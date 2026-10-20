import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch
from omegaconf import OmegaConf
from PIL import Image

from config.config import Config
from hexplane.dataloader.dnerf_dataset import DNerfDataset
from hexplane.depth import masked_depth_mse, render_depth
from hexplane.model import init_model
from hexplane.render.render import evaluation
from hexplane.render.trainer import Trainer


class Writer:
    def __init__(self):
        self.scalars = []

    def add_scalar(self, *args, **kwargs):
        self.scalars.append((args[0], float(args[1])))

    def flush(self):
        pass


class DepthSupervisionTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        for split in ("train", "test"):
            (self.root / split).mkdir()
            (self.root / "depth" / split).mkdir(parents=True)
            frames = []
            for i in range(2):
                image = np.full((8, 8, 4), 128, dtype=np.uint8)
                image[..., 3] = 255
                image[0, 0, 3] = 0
                Image.fromarray(image).save(self.root / split / f"r_{i}.png")
                np.save(self.root / "depth" / split / f"r_{i}.npy", np.full((8, 8), 3 + i * 0.1, np.float32))
                pose = np.eye(4).tolist()
                pose[2][3] = 3.0
                frames.append({"file_path": f"{split}/r_{i}", "transform_matrix": pose, "time": i})
            (self.root / f"transforms_{split}.json").write_text(
                json.dumps({"camera_angle_x": 0.7, "frames": frames}), encoding="utf-8"
            )

    def dataset(self, split="train", stack=False, **kwargs):
        return DNerfDataset(
            str(self.root), split=split, downsample=200, is_stack=stack,
            N_random_pose=1, depth_dir=f"depth/{split}", **kwargs
        )

    def cfg(self):
        cfg = OmegaConf.structured(Config())
        cfg.optim.n_iters = 2
        cfg.optim.batch_size = 4
        cfg.model.N_voxel_init = 125
        cfg.model.time_grid_init = 2
        cfg.model.time_grid_final = 2
        cfg.model.density_n_comp = [2, 2, 2]
        cfg.model.app_n_comp = [2, 2, 2]
        cfg.model.AppMode = "RGB"
        cfg.model.app_dim = 3
        cfg.model.upsample_list = []
        cfg.model.update_emptymask_list = []
        cfg.model.depth_loss = True
        cfg.model.TV_weight_density = 0
        cfg.model.TV_weight_app = 0
        return cfg

    def test_render_depth_background_and_gradients(self):
        weights = torch.tensor([[0.2, 0.3], [0.0, 0.0]], requires_grad=True)
        positions = torch.tensor([[2.0, 4.0]])
        depth = render_depth(weights, positions, 6.0)
        torch.testing.assert_close(depth, torch.tensor([4.6, 6.0]))
        depth.sum().backward()
        torch.testing.assert_close(weights.grad, torch.tensor([[-4.0, -2.0], [-4.0, -2.0]]))
        torch.testing.assert_close(
            render_depth(weights, positions, 6.0, "conditional"), torch.tensor([3.2, 6.0])
        )

    def test_loss_masks_before_arithmetic_and_rejects_wrong_sizes(self):
        prediction = torch.tensor([2.0, 3.0, 4.0, 5.0], requires_grad=True)
        target = torch.tensor([[3.0], [float("nan")], [float("inf")], [-1.0]])
        loss = masked_depth_mse(prediction, target)
        self.assertEqual(loss.item(), 1.0)
        loss.backward()
        torch.testing.assert_close(prediction.grad, torch.tensor([-2.0, 0.0, 0.0, 0.0]))
        zero = masked_depth_mse(prediction, torch.zeros(4))
        self.assertEqual(zero.item(), 0.0)
        self.assertTrue(zero.requires_grad)
        with self.assertRaises(ValueError):
            masked_depth_mse(prediction, torch.ones(3))
        with self.assertRaises(ValueError):
            masked_depth_mse(torch.tensor([float("nan")]), torch.ones(1))

    def test_dataset_conversion_resize_and_invalid_pixels(self):
        values = np.full((8, 8), 3000, np.float32)
        values[0, 2] = np.nan
        values[2, 0] = np.inf
        values[2, 2] = -1
        for i in range(2):
            np.save(self.root / "depth/train" / f"r_{i}.npy", values)
        dataset = self.dataset(stack=True, depth_type="camera_z", depth_scale=0.001)
        depth = dataset[0]["depths"].reshape(4, 4)
        self.assertEqual(tuple(dataset.all_depths.shape), (2, 4, 4, 1))
        self.assertEqual(depth[0, 0].item(), 0)  # Transparent RGB background.
        self.assertEqual(depth[0, 1].item(), 0)
        self.assertEqual(depth[1, 0].item(), 0)
        self.assertEqual(depth[1, 1].item(), 0)
        self.assertAlmostEqual(depth[2, 2].item(), 3 / abs(dataset.directions[2, 2, 2].item()), places=5)
        Image.fromarray(np.full((8, 8), 3000, np.uint16)).save(self.root / "depth/train/r_0.png")
        self.assertEqual(dataset.read_depth(self.root / "depth/train/r_0.png")[0, 0], 3000)
        pfm = self.root / "depth/train/r_0.pfm"
        with pfm.open("wb") as stream:
            stream.write(b"Pf\n8 8\n-1.0\n")
            stream.write(np.full((8, 8), 3.0, dtype="<f4").tobytes())
        np.testing.assert_array_equal(dataset.read_depth(pfm), np.full((8, 8), 3.0))
        with self.assertRaises(FileNotFoundError):
            self.dataset(depth_file_pattern="missing_{stem}.npy")

    def test_training_only_depth_and_image_sampler(self):
        cfg = self.cfg()
        train = self.dataset()
        test = DNerfDataset(str(self.root), split="test", downsample=200, is_stack=True)
        model, reso = init_model(cfg, train.scene_bbox, train.near_far, "cpu")
        trainer = Trainer(model, cfg, reso, train, test, Writer(), str(self.root / "run"), "cpu")
        trainer.init_sampler(train)
        rays, rgb, times, depths = trainer.sample_data(train, 0)
        self.assertEqual(tuple(depths.shape), (4, 1))
        self.assertTrue(trainer.depth_data)  # Test set has no depth, training set does.
        with self.assertRaisesRegex(ValueError, "no reference depths"):
            Trainer(model, cfg, reso, test, test, Writer(), str(self.root), "cpu")
        cfg.data.datasampler_type = "images"
        stacked = self.dataset(stack=True)
        trainer.cfg = cfg
        trainer.init_sampler(stacked)
        rays, _, times, depths = trainer.sample_data(stacked, 0)
        self.assertEqual(tuple(depths.shape), (4, 1))
        frame = int((times[0, 0].item() + 1) / 2)
        for ray, depth in zip(rays, depths):
            pixel = torch.isclose(stacked.all_rays[frame], ray).all(dim=-1).nonzero()[0, 0]
            torch.testing.assert_close(depth, stacked.all_depths[frame].reshape(-1, 1)[pixel])

    def test_two_iteration_training_with_depth(self):
        cfg = self.cfg()
        train = self.dataset()
        test = DNerfDataset(str(self.root), split="test", downsample=200, is_stack=True)
        model, reso = init_model(cfg, train.scene_bbox, train.near_far, "cpu")
        output = self.root / "run"
        writer = Writer()
        Trainer(model, cfg, reso, train, test, writer, str(output), "cpu").train()
        stats = json.loads((output / "training_stats.json").read_text())
        self.assertEqual(stats["iterations"], 2)
        self.assertGreater(stats["elapsed_seconds"], 0)
        mse_values = [value for tag, value in writer.scalars if tag == "train/mse"]
        self.assertEqual(len(mse_values), 2)
        self.assertTrue(all(0 <= value <= 1 for value in mse_values))

    def test_hierarchical_depth_matches_selected_pixels(self):
        cfg = self.cfg()
        cfg.data.datasampler_type = "hierach"
        cfg.data.key_f_num = 1
        cfg.data.stage_1_iteration = 1
        cfg.data.stage_2_iteration = 1
        dataset = SimpleNamespace(
            depth_data=True, ndc_ray=False,
            all_rgbs=torch.rand(1, 3, 8, 3),
            all_rays=torch.arange(48).reshape(1, 8, 6).float(),
            all_times=torch.tensor([[0.0, 1.0, 2.0]]),
            all_depths=torch.stack([torch.arange(8) + 10 * i for i in range(3)]).reshape(1, 3, 8, 1).float(),
            global_mean_rgb=torch.zeros(1, 8, 3),
        )
        trainer = Trainer(SimpleNamespace(), cfg, [5]*3, dataset, dataset, Writer(), str(self.root), "cpu")
        trainer.init_sampler(dataset)
        for iteration in (0, 2, 3):
            rays, _, times, depths = trainer.sample_data(dataset, iteration)
            expected = rays[:, 0] / 6 + times[:, 0] * 10
            torch.testing.assert_close(depths[:, 0], expected)

    def test_model_depth_gradients_and_checkpoint_settings(self):
        cfg = self.cfg()
        dataset = self.dataset()
        model, _ = init_model(cfg, dataset.scene_bbox, dataset.near_far, "cpu")
        rays = torch.tensor([[0., 0., 3., 0., 0., -1.]])
        _, depth, _, _ = model(rays, torch.zeros(1, 1), N_samples=8)
        self.assertTrue(depth.requires_grad)
        depth.sum().backward()
        self.assertTrue(any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.density_plane))
        model.depth_loss = False
        self.assertFalse(model(rays, torch.zeros(1, 1), N_samples=8)[1].requires_grad)
        cfg.systems.ckpt = "old.th"
        cfg.model.depth_loss_weight = 0
        cfg.model.depth_render_mode = "conditional"
        with patch("hexplane.model.torch.load", return_value=model):
            restored, _ = init_model(cfg, dataset.scene_bbox, dataset.near_far, "cpu")
        self.assertFalse(restored.depth_loss)
        self.assertEqual(restored.depth_render_mode, "conditional")

    def test_evaluation_raw_depth_and_pixel_weighted_metrics(self):
        cfg = self.cfg()
        dataset = self.dataset(split="test", stack=True)
        output = self.root / "evaluation"
        def renderer(rays, times, model, **kwargs):
            size = rays.shape[0]
            return torch.full((size, 3), 0.4), None, torch.full((size,), 4.0), None, None
        with patch("hexplane.render.render.OctreeRender_trilinear_fast", side_effect=renderer), patch("hexplane.render.render.imageio.mimwrite"):
            evaluation(dataset, SimpleNamespace(depth_render_mode="far"), cfg, str(output), N_vis=-1, compute_extra_metrics=False, device="cpu")
        metrics = json.loads((output / "depth_metrics.json").read_text())
        self.assertEqual(metrics["valid_pixels"], 30)
        self.assertAlmostEqual(metrics["mae"], 0.95, places=5)
        self.assertAlmostEqual(metrics["rmse"], (0.905)**0.5, places=5)
        raw = np.load(output / "depth_raw/000_depth.npy")
        self.assertEqual(raw.dtype, np.float32)
        np.testing.assert_array_equal(raw, np.full((4, 4), 4.0))

    def test_rgb_only_and_unsupported_depth_fail_early(self):
        cfg = self.cfg()
        cfg.model.depth_loss = False
        dataset = DNerfDataset(str(self.root), split="test", downsample=200, is_stack=True)
        self.assertFalse(dataset.depth_data)
        self.assertNotIn("depths", dataset[0])
        model, _ = init_model(cfg, dataset.scene_bbox, dataset.near_far, "cpu")
        rays = dataset[0]["rays"]
        times = dataset[0]["time"]
        rgb_far = model(rays, times, N_samples=8)[0]
        model.depth_render_mode = "conditional"
        torch.testing.assert_close(rgb_far, model(rays, times, N_samples=8)[0])
        ndc = SimpleNamespace(depth_data=True, ndc_ray=True)
        cfg.model.depth_loss = True
        with self.assertRaisesRegex(ValueError, "NDC"):
            Trainer(model, cfg, [5]*3, ndc, ndc, Writer(), str(self.root), "cpu")

    def test_evaluation_without_reference_or_raw_output(self):
        cfg = self.cfg()
        cfg.save_depth_raw = False
        dataset = DNerfDataset(str(self.root), split="test", downsample=200, is_stack=True)
        output = self.root / "no_depth_evaluation"
        def renderer(rays, times, model, **kwargs):
            size = rays.shape[0]
            return torch.full((size, 3), 0.4), None, torch.full((size,), 4.0), None, None
        with patch("hexplane.render.render.OctreeRender_trilinear_fast", side_effect=renderer), patch("hexplane.render.render.imageio.mimwrite"):
            evaluation(dataset, SimpleNamespace(), cfg, str(output), N_vis=-1, compute_extra_metrics=False, device="cpu")
        self.assertTrue((output / "000.png").is_file())
        self.assertFalse((output / "depth_raw").exists())
        self.assertFalse((output / "depth_metrics.json").exists())


if __name__ == "__main__":
    unittest.main()
