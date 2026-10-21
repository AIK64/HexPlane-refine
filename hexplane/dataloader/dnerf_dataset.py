import json
import os
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms as T
from tqdm import tqdm

from .ray_utils import get_ray_directions_blender, get_rays, read_pfm

blender2opencv = torch.Tensor([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])


def trans_t(t):
    return torch.Tensor(
        [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, t], [0, 0, 0, 1]]
    ).float()


def rot_phi(phi):
    return torch.Tensor(
        [
            [1, 0, 0, 0],
            [0, np.cos(phi), -np.sin(phi), 0],
            [0, np.sin(phi), np.cos(phi), 0],
            [0, 0, 0, 1],
        ]
    ).float()


def rot_theta(th):
    return torch.Tensor(
        [
            [np.cos(th), 0, -np.sin(th), 0],
            [0, 1, 0, 0],
            [np.sin(th), 0, np.cos(th), 0],
            [0, 0, 0, 1],
        ]
    ).float()


def pose_spherical(theta, phi, radius):
    c2w = trans_t(radius)
    c2w = rot_phi(phi / 180.0 * np.pi) @ c2w
    c2w = rot_theta(theta / 180.0 * np.pi) @ c2w
    c2w = (
        torch.Tensor(
            np.array([[-1, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0], [0, 0, 0, 1]])
        )
        @ c2w
        @ blender2opencv
    )
    return c2w


class DNerfDataset(Dataset):
    def __init__(
        self,
        datadir,
        split="train",
        downsample=2.0,
        is_stack=False,
        cal_fine_bbox=False,
        N_vis=-1,
        time_scale=1.0,
        scene_bbox_min=[-1.0, -1.0, -1.0],
        scene_bbox_max=[1.0, 1.0, 1.0],
        N_random_pose=1000,
        depth_dir=None,
        depth_file_pattern="{stem}.npy",
        depth_type="ray_distance",
        depth_scale=1.0,
    ):
        self.root_dir = datadir
        self.split = split
        self.downsample = downsample
        self.img_wh = (int(800 / downsample), int(800 / downsample))
        self.is_stack = is_stack
        self.N_vis = N_vis  # evaluate images for every N_vis images
        self.depth_data = depth_dir is not None
        self.depth_dir = None
        if self.depth_data:
            self.depth_dir = Path(depth_dir)
            if not self.depth_dir.is_absolute():
                self.depth_dir = Path(datadir) / self.depth_dir
            if not self.depth_dir.is_dir():
                raise FileNotFoundError(f"Reference depth directory does not exist: {self.depth_dir}")
        if depth_type not in ("ray_distance", "camera_z", "relative_camera_z"):
            raise ValueError("depth_type must be ray_distance, camera_z or relative_camera_z.")
        if not np.isfinite(depth_scale) or depth_scale <= 0:
            raise ValueError("depth_scale must be finite and positive.")
        self.depth_file_pattern = depth_file_pattern
        self.depth_type = depth_type
        self.depth_scale = depth_scale

        self.time_scale = time_scale
        self.world_bound_scale = 1.1

        self.near = 2.0
        self.far = 6.0
        self.near_far = [2.0, 6.0]

        self.define_transforms()  # transform to torch.Tensor

        self.scene_bbox = torch.tensor([scene_bbox_min, scene_bbox_max])
        self.blender2opencv = np.array(
            [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]
        )

        self.read_meta()  # Read meta data

        # Calculate a more fine bbox based on near and far values of each ray.
        if cal_fine_bbox:
            xyz_min, xyz_max = self.compute_bbox()
            self.scene_bbox = torch.stack((xyz_min, xyz_max), dim=0)

        self.define_proj_mat()

        self.white_bg = True
        self.ndc_ray = False

        self.N_random_pose = N_random_pose
        self.center = torch.mean(self.scene_bbox, axis=0).float().view(1, 1, 3)
        self.radius = (self.scene_bbox[1] - self.center).float().view(1, 1, 3)
        # Generate N_random_pose random poses, which we could render depths from these poses and apply depth smooth loss to the rendered depth.
        if split == "train":
            self.init_random_pose()

    def init_random_pose(self):
        # Randomly sample N_random_pose radius, phi, theta and times.
        radius = np.random.randn(self.N_random_pose) * 0.1 + 4
        phi = np.random.rand(self.N_random_pose) * 360 - 180
        theta = np.random.rand(self.N_random_pose) * 360 - 180
        random_times = self.time_scale * (torch.rand(self.N_random_pose) * 2.0 - 1.0)
        self.random_times = random_times

        # Generate rays from random radius, phi, theta and times.
        self.random_rays = []
        for i in range(self.N_random_pose):
            random_poses = pose_spherical(theta[i], phi[i], radius[i])
            rays_o, rays_d = get_rays(self.directions, random_poses)
            self.random_rays += [torch.cat([rays_o, rays_d], 1)]

        self.random_rays = torch.stack(self.random_rays, 0).reshape(
            -1, *self.img_wh[::-1], 6
        )

    def compute_bbox(self):
        print("compute_bbox_by_cam_frustrm: start")
        xyz_min = torch.Tensor([np.inf, np.inf, np.inf])
        xyz_max = -xyz_min
        rays_o = self.all_rays[:, 0:3]
        viewdirs = self.all_rays[:, 3:6]
        pts_nf = torch.stack(
            [rays_o + viewdirs * self.near, rays_o + viewdirs * self.far]
        )
        xyz_min = torch.minimum(xyz_min, pts_nf.amin((0, 1)))
        xyz_max = torch.maximum(xyz_max, pts_nf.amax((0, 1)))
        print("compute_bbox_by_cam_frustrm: xyz_min", xyz_min)
        print("compute_bbox_by_cam_frustrm: xyz_max", xyz_max)
        print("compute_bbox_by_cam_frustrm: finish")
        xyz_shift = (xyz_max - xyz_min) * (self.world_bound_scale - 1) / 2
        xyz_min -= xyz_shift
        xyz_max += xyz_shift
        return xyz_min, xyz_max

    def read_depth(self, filename):
        filename = Path(filename)
        if filename.suffix.lower() == ".npy":
            depth = np.load(filename, allow_pickle=False)
        elif filename.suffix.lower() == ".pfm":
            depth = read_pfm(str(filename))[0]
        elif filename.suffix.lower() in (".png", ".tif", ".tiff"):
            with Image.open(filename) as image:
                depth = np.array(image)
        else:
            raise ValueError(f"Unsupported depth format: {filename.suffix}")
        depth = np.asarray(depth, dtype=np.float32)
        if depth.ndim != 2:
            raise ValueError(f"Depth must be a scalar H x W image: {filename}, {depth.shape}")
        return depth

    def load_frame_depth(self, frame, original_wh, alpha):
        frame_path = Path(frame["file_path"])
        relative_path = self.depth_file_pattern.format(
            stem=frame_path.stem,
            frame=frame_path.with_suffix("").as_posix(),
            split=self.split,
        )
        filename = self.depth_dir / relative_path
        if not filename.is_file():
            raise FileNotFoundError(f"Missing reference depth for {frame['file_path']}: {filename}")
        depth = torch.from_numpy(np.ascontiguousarray(self.read_depth(filename)))
        if tuple(depth.shape) != (original_wh[1], original_wh[0]):
            raise ValueError(
                f"Depth and original RGB resolution must match: {filename}, "
                f"depth {tuple(depth.shape)}, RGB {original_wh[::-1]}"
            )
        depth = depth * self.depth_scale
        depth = torch.where(torch.isfinite(depth) & (depth > 0), depth, 0)
        # Nearest sampling avoids blending valid depths across invalid pixels or edges.
        depth = F.interpolate(
            depth[None, None], size=self.img_wh[::-1], mode="nearest"
        )[0, 0]
        alpha = F.interpolate(
            alpha.float()[None, None], size=self.img_wh[::-1], mode="nearest"
        )[0, 0]
        if self.depth_type == "camera_z":
            depth = depth / self.directions[..., 2].abs().clamp_min(1e-8)
        depth = depth.reshape(-1, 1)
        valid = torch.isfinite(depth) & (depth > 0) & (alpha.reshape(-1, 1) > 0)
        if self.depth_type != "relative_camera_z":
            valid &= (depth >= self.near) & (depth <= self.far)
        return torch.where(valid, depth, 0)

    def read_meta(self):
        with open(os.path.join(self.root_dir, f"transforms_{self.split}.json")) as f:
            self.meta = json.load(f)

        w, h = self.img_wh
        self.focal = (
            0.5 * 800 / np.tan(0.5 * self.meta["camera_angle_x"])
        )  # original focal length
        self.focal *= (
            self.img_wh[0] / 800
        )  # modify focal length to match size self.img_wh

        # ray directions for all pixels, same for all images (same H, W, focal)
        self.directions = get_ray_directions_blender(
            h, w, [self.focal, self.focal]
        )  # (h, w, 3)
        self.directions = self.directions / torch.norm(
            self.directions, dim=-1, keepdim=True
        )
        self.intrinsics = torch.tensor(
            [[self.focal, 0, w / 2], [0, self.focal, h / 2], [0, 0, 1]]
        ).float()

        self.image_paths = []
        self.poses = []
        self.all_rays = []
        self.all_times = []
        self.all_rgbs = []
        self.all_depths = [] if self.depth_data else None

        img_eval_interval = (
            1 if self.N_vis <= 0 else max(len(self.meta["frames"]) // self.N_vis, 1)
        )
        idxs = list(range(0, len(self.meta["frames"]), img_eval_interval))
        for i in tqdm(
            idxs, desc=f"Loading data {self.split} ({len(idxs)})"
        ):  # img_list:#
            frame = self.meta["frames"][i]
            pose = np.array(frame["transform_matrix"]) @ self.blender2opencv
            c2w = torch.FloatTensor(pose)
            self.poses += [c2w]

            image_path = os.path.join(self.root_dir, f"{frame['file_path']}.png")
            self.image_paths += [image_path]
            img = Image.open(image_path)
            original_wh = img.size
            original_alpha = (
                torch.from_numpy(np.array(img.getchannel("A"))) if self.depth_data else None
            )

            if self.downsample != 1.0:
                img = img.resize(self.img_wh, Image.LANCZOS)
            img = self.transform(img)  # (4, h, w)
            img = img.view(4, -1).permute(1, 0)  # (h*w, 4) RGBA
            if self.depth_data:
                self.all_depths.append(self.load_frame_depth(frame, original_wh, original_alpha))
            img = img[:, :3] * img[:, -1:] + (
                1 - img[:, -1:]
            )  # blend A to RGB, white background
            self.all_rgbs += [img]

            rays_o, rays_d = get_rays(self.directions, c2w)  # Get rays, both (h*w, 3).
            self.all_rays += [torch.cat([rays_o, rays_d], 1)]  # (h*w, 6)
            cur_time = torch.tensor(
                frame["time"]
                if "time" in frame
                else float(i) / (len(self.meta["frames"]) - 1)
            ).expand(rays_o.shape[0], 1)
            self.all_times += [cur_time]

        self.poses = torch.stack(self.poses)
        #  self.is_stack stacks all images into a big chunk, with shape (N, H, W, 3).
        #  Otherwise, all images are kept as a set of rays with shape (N_s, 3), where N_s = H * W * N
        if not self.is_stack:
            self.all_rays = torch.cat(
                self.all_rays, 0
            )  # (len(self.meta['frames])*h*w, 3)
            self.all_rgbs = torch.cat(
                self.all_rgbs, 0
            )  # (len(self.meta['frames])*h*w, 3)
            self.all_times = torch.cat(self.all_times, 0)
            if self.depth_data:
                self.all_depths = torch.cat(self.all_depths, 0)

        else:
            self.all_rays = torch.stack(
                self.all_rays, 0
            )  # (len(self.meta['frames]),h*w, 3)
            self.all_rgbs = torch.stack(self.all_rgbs, 0).reshape(
                -1, *self.img_wh[::-1], 3
            )  # (len(self.meta['frames]),h,w,3)
            self.all_times = torch.stack(self.all_times, 0)
            if self.depth_data:
                self.all_depths = torch.stack(self.all_depths, 0).reshape(
                    -1, *self.img_wh[::-1], 1
                )

        if self.depth_data and not (self.all_depths > 0).any():
            raise ValueError(
                "No valid reference depths remain. Check depth_type, depth_scale, "
                "foreground coverage and the scene near/far range."
            )

        self.all_times = self.time_scale * (self.all_times * 2.0 - 1.0)

    def define_transforms(self):
        self.transform = T.ToTensor()

    def define_proj_mat(self):
        self.proj_mat = self.intrinsics.unsqueeze(0) @ torch.inverse(self.poses)[:, :3]

    def world2ndc(self, points, lindisp=None):
        device = points.device
        return (points - self.center.to(device)) / self.radius.to(device)

    def __len__(self):
        return len(self.all_rgbs)

    def get_val_pose(self):
        """
        Get validation poses and times (NeRF-like rotating cameras poses).
        """
        render_poses = torch.stack(
            [
                pose_spherical(angle, -30.0, 4.0)
                for angle in np.linspace(-180, 180, 40 + 1)[:-1]
            ],
            0,
        )
        render_times = torch.linspace(0.0, 1.0, render_poses.shape[0]) * 2.0 - 1.0
        return render_poses, self.time_scale * render_times

    def get_val_rays(self):
        """
        Get validation rays and times (NeRF-like rotating cameras poses).
        """
        val_poses, val_times = self.get_val_pose()  # get valitdation poses and times
        rays_all = []  # initialize list to store [rays_o, rays_d]

        for i in range(val_poses.shape[0]):
            c2w = torch.FloatTensor(val_poses[i])
            rays_o, rays_d = get_rays(self.directions, c2w)  # both (h*w, 3)
            rays = torch.cat([rays_o, rays_d], 1)  # (h*w, 6)
            rays_all.append(rays)
        return rays_all, torch.FloatTensor(val_times)

    def __getitem__(self, idx):
        if self.split == "train":  # use data in the buffers
            sample = {
                "rays": self.all_rays[idx],
                "rgbs": self.all_rgbs[idx],
                "time": self.all_times[idx],
            }
        else:  # create data for each image separately
            img = self.all_rgbs[idx]
            rays = self.all_rays[idx]
            time = self.all_times[idx]
            sample = {"rays": rays, "rgbs": img, "time": time}
        if self.depth_data:
            sample["depths"] = self.all_depths[idx]
        return sample

    def get_random_pose(self, batch_size, patch_size, batching="all_images"):
        """
        Apply Geometry Regularization from RegNeRF.
        This function randomly samples many patches from random poses.
        """
        n_patches = batch_size // (patch_size**2)

        N_random = self.random_rays.shape[0]
        # Sample images
        if batching == "all_images":
            idx_img = np.random.randint(0, N_random, size=(n_patches, 1))
        elif batching == "single_image":
            idx_img = np.random.randint(0, N_random)
            idx_img = np.full((n_patches, 1), idx_img, dtype=np.int)
        else:
            raise ValueError("Not supported batching type!")
        idx_img = torch.Tensor(idx_img).long()
        H, W = self.random_rays[0].shape[0], self.random_rays[0].shape[1]
        # Sample start locations
        x0 = np.random.randint(
            int(W // 4), int(W // 4 * 3) - patch_size + 1, size=(n_patches, 1, 1)
        )
        y0 = np.random.randint(
            int(H // 4), int(H // 4 * 3) - patch_size + 1, size=(n_patches, 1, 1)
        )
        xy0 = np.concatenate([x0, y0], axis=-1)
        patch_idx = xy0 + np.stack(
            np.meshgrid(np.arange(patch_size), np.arange(patch_size), indexing="xy"),
            axis=-1,
        ).reshape(1, -1, 2)

        patch_idx = torch.Tensor(patch_idx).long()
        # Subsample images
        out = self.random_rays[idx_img, patch_idx[..., 1], patch_idx[..., 0]]

        return out, self.random_times[idx_img]
