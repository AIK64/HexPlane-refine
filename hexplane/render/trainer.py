import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch
from tqdm.auto import tqdm
from hexplane.depth import masked_depth_mse

from hexplane.render.render import OctreeRender_trilinear_fast as renderer
from hexplane.render.render import evaluation
from hexplane.render.util.Reg import TVLoss, compute_dist_loss
from hexplane.render.util.Sampling import GM_Resi, cal_n_samples
from hexplane.render.util.util import N_to_reso


class SimpleSampler:
    """
    A sampler that samples a batch of ids randomly.
    """

    def __init__(self, total, batch):
        self.total = total
        self.batch = batch
        self.curr = total
        self.ids = None

    def nextids(self):
        self.curr += self.batch
        if self.curr + self.batch > self.total:
            self.ids = torch.LongTensor(np.random.permutation(self.total))
            self.curr = 0
        return self.ids[self.curr : self.curr + self.batch]


class Trainer:
    def __init__(
        self,
        model,
        cfg,
        reso_cur,
        train_dataset,
        test_dataset,
        summary_writer,
        logfolder,
        device,
    ):
        self.model = model
        self.cfg = cfg
        self.reso_cur = reso_cur
        self.train_dataset = train_dataset
        self.test_dataset = test_dataset
        self.summary_writer = summary_writer
        self.logfolder = logfolder
        self.device = device
        self.depth_data = bool(getattr(train_dataset, "depth_data", False))
        if cfg.model.depth_loss_weight < 0:
            raise ValueError("depth_loss_weight must be non-negative.")
        self.depth_supervision = cfg.model.depth_loss and cfg.model.depth_loss_weight > 0
        if self.depth_supervision and not self.depth_data:
            raise ValueError(
                "Depth loss is enabled but the training dataset has no reference depths. "
                "For DNeRF, configure data.depth_train_dir and the depth format/scale."
            )
        if self.depth_supervision and train_dataset.ndc_ray:
            raise ValueError(
                "Metric depth supervision for neural3D_NDC is not supported: "
                "reference distances must first be converted to NDC ray parameters."
            )
        self.model.depth_loss = self.depth_supervision
        self.model.depth_render_mode = cfg.model.depth_render_mode

    def get_lr_decay_factor(self, step):
        """
        Calculate the learning rate decay factor = current_lr / initial_lr.
        """
        if self.cfg.optim.lr_decay_step == -1:
            self.cfg.optim.lr_decay_step = self.cfg.optim.n_iters

        if self.cfg.optim.lr_decay_type == "exp":  # exponential decay
            lr_factor = self.cfg.optim.lr_decay_target_ratio ** (
                step / self.cfg.optim.lr_decay_step
            )
        elif self.cfg.optim.lr_decay_type == "linear":  # linear decay
            lr_factor = self.cfg.optim.lr_decay_target_ratio + (
                1 - self.cfg.optim.lr_decay_target_ratio
            ) * (1 - step / self.cfg.optim.lr_decay_step)
        elif self.cfg.optim.lr_decay_type == "cosine":  # consine decay
            lr_factor = self.cfg.optim.lr_decay_target_ratio + (
                1 - self.cfg.optim.lr_decay_target_ratio
            ) * 0.5 * (1 + math.cos(math.pi * step / self.cfg.optim.lr_decay_step))

        return lr_factor

    def get_voxel_upsample_list(self):
        """
        Precompute  spatial and temporal grid upsampling sizes.
        """
        upsample_list = self.cfg.model.upsample_list
        if (
            self.cfg.model.upsampling_type == "unaligned"
        ):  # logaritmic upsampling. See explation of "unaligned" in model/__init__.py.
            N_voxel_list = (
                torch.round(
                    torch.exp(
                        torch.linspace(
                            np.log(self.cfg.model.N_voxel_init),
                            np.log(self.cfg.model.N_voxel_final),
                            len(upsample_list) + 1,
                        )
                    )
                ).long()
            ).tolist()[1:]
        elif (
            self.cfg.model.upsampling_type == "aligned"
        ):  # aligned upsampling doesn't need precompute N_voxel_list.
            N_voxel_list = None
        # logaritmic upsampling for time grid.
        Time_grid_list = (
            torch.round(
                torch.exp(
                    torch.linspace(
                        np.log(self.cfg.model.time_grid_init),
                        np.log(self.cfg.model.time_grid_final),
                        len(upsample_list) + 1,
                    )
                )
            ).long()
        ).tolist()[1:]
        self.N_voxel_list = N_voxel_list
        self.Time_grid_list = Time_grid_list

    def sample_data(self, train_dataset, iteration):
        """
        Sample a batch of data from the dataset.
        """
        train_depth = None
        # sample rays: shuffle all the rays of training dataset and sampled a batch of rays from them.
        if self.cfg.data.datasampler_type == "rays":
            ray_idx = self.sampler.nextids()
            data = train_dataset[ray_idx]
            rays_train, rgb_train, frame_time = (
                data["rays"],
                data["rgbs"].to(self.device),
                data["time"],
            )
            if self.depth_data:
                train_depth = data["depths"].to(self.device).reshape(-1, 1)
        # sample images: randomly pick one image from the training dataset and sample a batch of rays from all the rays of the image.
        elif self.cfg.data.datasampler_type == "images":
            img_i = self.sampler.nextids()
            data = train_dataset[img_i]
            rays_train, rgb_train, frame_time = (
                data["rays"].reshape(-1, 6),
                data["rgbs"].to(self.device).view(-1, 3),
                data["time"].reshape(-1, 1),
            )
            select_inds = torch.randperm(rays_train.shape[0])[
                : self.cfg.optim.batch_size
            ]
            rays_train = rays_train[select_inds]
            rgb_train = rgb_train[select_inds]
            frame_time = frame_time[select_inds]
            if self.depth_data:
                train_depth = data["depths"].to(self.device).view(-1, 1)[select_inds]
        # hierarchical sampling from dyNeRF: hierachical sampling involves three stages of samplings.
        elif self.cfg.data.datasampler_type == "hierach":
            # Stage 1: randomly sample a single image from an arbitrary camera.
            # And sample a batch of rays from all the rays of the image based on the difference of global median and local values.
            # Stage 1 only samples key-frames, which is the frame every self.cfg.data.key_f_num frames.
            if iteration <= self.cfg.data.stage_1_iteration:
                cam_i = np.random.choice(train_dataset.all_rgbs.shape[0])
                index_i = np.random.choice(
                    train_dataset.all_rgbs.shape[1] // self.cfg.data.key_f_num
                )
                rgb_train = (
                    train_dataset.all_rgbs[cam_i, index_i * self.cfg.data.key_f_num]
                    .view(-1, 3)
                    .to(self.device)
                )
                rays_train = train_dataset.all_rays[cam_i].view(-1, 6)
                frame_time = train_dataset.all_times[
                    cam_i, index_i * self.cfg.data.key_f_num
                ]
                depth_index = index_i * self.cfg.data.key_f_num
                # Calcualte the probability of sampling each ray based on the difference of global median and local values.
                probability = GM_Resi(
                    rgb_train, self.global_mean[cam_i], self.cfg.data.stage_1_gamma
                )
                select_inds = torch.multinomial(
                    probability, self.cfg.optim.batch_size
                ).to(rays_train.device)
                rays_train = rays_train[select_inds]
                rgb_train = rgb_train[select_inds]
                frame_time = torch.ones_like(rays_train[:, 0:1]) * frame_time
            elif (
                iteration
                <= self.cfg.data.stage_2_iteration + self.cfg.data.stage_1_iteration
            ):
                # Stage 2: basically the same as stage 1, but samples all the frames instead of only key-frames.
                cam_i = np.random.choice(train_dataset.all_rgbs.shape[0])
                index_i = np.random.choice(train_dataset.all_rgbs.shape[1])
                rgb_train = (
                    train_dataset.all_rgbs[cam_i, index_i].view(-1, 3).to(self.device)
                )
                rays_train = train_dataset.all_rays[cam_i].view(-1, 6)
                frame_time = train_dataset.all_times[cam_i, index_i]
                depth_index = index_i
                probability = GM_Resi(
                    rgb_train, self.global_mean[cam_i], self.cfg.data.stage_2_gamma
                )
                select_inds = torch.multinomial(
                    probability, self.cfg.optim.batch_size
                ).to(rays_train.device)
                rays_train = rays_train[select_inds]
                rgb_train = rgb_train[select_inds]
                frame_time = torch.ones_like(rays_train[:, 0:1]) * frame_time
            else:
                # Stage 3: randomly sample one frame and sample a batch of rays from the sampled frame.
                # TO sample a batch of rays from this frame, we calcualate the value changes of rays compared to nearby timesteps, and sample based on the value changes.
                cam_i = np.random.choice(train_dataset.all_rgbs.shape[0])
                N_time = train_dataset.all_rgbs.shape[1]
                # Sample two adjacent time steps within a range of 25 frames.
                index_i = np.random.choice(N_time)
                index_2 = np.random.choice(
                    min(N_time, index_i + 25) - max(index_i - 25, 0)
                ) + max(index_i - 25, 0)
                rgb_train = (
                    train_dataset.all_rgbs[cam_i, index_i].view(-1, 3).to(self.device)
                )
                rgb_ref = (
                    train_dataset.all_rgbs[cam_i, index_2].view(-1, 3).to(self.device)
                )
                rays_train = train_dataset.all_rays[cam_i].view(-1, 6)
                frame_time = train_dataset.all_times[cam_i, index_i]
                depth_index = index_i
                # Calcualte the temporal difference between the two frames as sampling probability.
                probability = torch.clamp(
                    1 / 3 * torch.norm(rgb_train - rgb_ref, p=1, dim=-1),
                    min=self.cfg.data.stage_3_alpha,
                )
                select_inds = torch.multinomial(
                    probability, self.cfg.optim.batch_size
                ).to(rays_train.device)
                rays_train = rays_train[select_inds]
                rgb_train = rgb_train[select_inds]
                frame_time = torch.ones_like(rays_train[:, 0:1]) * frame_time
            if self.depth_data:
                depths = train_dataset.all_depths[cam_i, depth_index].reshape(-1, 1)
                train_depth = depths[select_inds.to(depths.device)].to(self.device)
        return rays_train, rgb_train, frame_time, train_depth

    def init_sampler(self, train_dataset):
        """
        Initialize the sampler for the training dataset.
        """
        if self.cfg.data.datasampler_type == "rays":
            self.sampler = SimpleSampler(len(train_dataset), self.cfg.optim.batch_size)
        elif self.cfg.data.datasampler_type == "images":
            self.sampler = SimpleSampler(len(train_dataset), 1)
        elif self.cfg.data.datasampler_type == "hierach":
            self.global_mean = train_dataset.global_mean_rgb.to(self.device)

    def train(self):
        torch.cuda.empty_cache()
        # 训练统计：同步 GPU 后开始计时，并重置本次训练的显存峰值。
        training_device = torch.device(self.device)
        uses_cuda = training_device.type == "cuda"
        if uses_cuda:
            torch.cuda.synchronize(training_device)
            torch.cuda.reset_peak_memory_stats(training_device)
        start_time = time.perf_counter()

        # load the training and testing dataset and other settings.
        train_dataset = self.train_dataset
        test_dataset = self.test_dataset
        model = self.model
        self.depth_data = bool(getattr(train_dataset, "depth_data", False))
        summary_writer = self.summary_writer
        reso_cur = self.reso_cur

        ndc_ray = train_dataset.ndc_ray  # if the rays are in NDC
        white_bg = test_dataset.white_bg  # if the background is white

        # Calculate the number of samples for each ray based on the current resolution.
        nSamples = min(
            self.cfg.model.nSamples,
            cal_n_samples(reso_cur, self.cfg.model.step_ratio),
        )

        # Filter the rays based on the bbox
        if (self.cfg.data.datasampler_type == "rays") and (ndc_ray is False):
            allrays, allrgbs, alltimes = (
                train_dataset.all_rays,
                train_dataset.all_rgbs,
                train_dataset.all_times,
            )
            if self.depth_data:
                alldepths = train_dataset.all_depths
            else:
                alldepths = None
            # 过滤掉不我们的aabb包围盒相交的光线
            allrays, allrgbs, alltimes, alldepths = model.filtering_rays(
                allrays, allrgbs, alltimes, alldepths, bbox_only=True
            )
            train_dataset.all_rays = allrays
            train_dataset.all_rgbs = allrgbs
            train_dataset.all_times = alltimes
            train_dataset.all_depths = alldepths

        # initialize the data sampler
        self.init_sampler(train_dataset)
        # precompute the voxel upsample list
        # 作用就是生成一组上采样的计划空间体素分辨率数组，以及时间分辨率数据
        # 比如N_voxel_list = [128000, 256000, 512000]  到达制定轮次的时候增加到指定的分辨率
        self.get_voxel_upsample_list()

        # Initialiaze TV loss on planse
        # Total Variation Loss（总变分损失） 主要实现奖励相邻特征之间的平滑
        tvreg_s = TVLoss()  # TV loss on the spatial planes
        tvreg_s_t = TVLoss(
            1.0, self.cfg.model.TV_t_s_ratio
        )  # TV loss on the spatial-temporal planes

        # 这里是创建了进度条
        pbar = tqdm(
            range(self.cfg.optim.n_iters),
            miniters=self.cfg.systems.progress_refresh_rate,
            file=sys.stdout,
        )

        PSNRs, PSNRs_test = [], [0]
        torch.cuda.empty_cache()

        # Initialize the optimizer
        # 取出模型中可供学习的参数  并且对于不同类别的参数进行了分组  
        # 比如特征平面参数  MLP参数  其他参数  这样对于不同组的参数可以设置不同的学习率
        grad_vars = model.get_optparam_groups(self.cfg.optim)
        optimizer = torch.optim.Adam(
            grad_vars, betas=(self.cfg.optim.beta1, self.cfg.optim.beta2)
        )

        for iteration in pbar:
            # Sample dat
            # 获取本轮的batchsize组对应的真值ray数据
            rays_train, rgb_train, frame_time, depth = self.sample_data(
                train_dataset, iteration
            )
            # Render the rgb values of rays
            # 使用模型计算估计值
            rgb_map, alphas_map, depth_map, weights, uncertainty = renderer(
                rays_train,
                frame_time,
                model,
                chunk=self.cfg.optim.batch_size,
                N_samples=nSamples,
                white_bg=white_bg,
                ndc_ray=ndc_ray,
                device=self.device,
                is_train=True,
            )

            # Calculate the loss
            loss = torch.mean((rgb_map - rgb_train) ** 2)
            total_loss = loss

            # Calculate the learning rate decay factor
            lr_factor = self.get_lr_decay_factor(iteration)

            # regularization
            # TV loss on the density planes
            if self.cfg.model.TV_weight_density > 0:
                TV_weight_density = lr_factor * self.cfg.model.TV_weight_density
                loss_tv = model.TV_loss_density(tvreg_s, tvreg_s_t) * TV_weight_density
                total_loss = total_loss + loss_tv
                summary_writer.add_scalar(
                    "train/reg_tv_density",
                    loss_tv.detach().item(),
                    global_step=iteration,
                )

            # TV loss on the appearance planes
            if self.cfg.model.TV_weight_app > 0:
                TV_weight_app = lr_factor * self.cfg.model.TV_weight_app
                loss_tv = model.TV_loss_app(tvreg_s, tvreg_s_t) * TV_weight_app
                total_loss = total_loss + loss_tv
                summary_writer.add_scalar(
                    "train/reg_tv_app", loss_tv.detach().item(), global_step=iteration
                )

            # L1 loss on the density planes
            if self.cfg.model.L1_weight_density > 0:
                L1_weight_density = lr_factor * self.cfg.model.L1_weight_density
                loss_l1 = model.L1_loss_density() * L1_weight_density
                total_loss = total_loss + loss_l1
                summary_writer.add_scalar(
                    "train/reg_l1_density",
                    loss_l1.detach().item(),
                    global_step=iteration,
                )

            # L1 loss on the appearance planes
            if self.cfg.model.L1_weight_app > 0:
                L1_weight_app = lr_factor * self.cfg.model.L1_weight_app
                loss_l1 = model.L1_loss_app() * L1_weight_app
                total_loss = total_loss + loss_l1
                summary_writer.add_scalar(
                    "train/reg_l1_app", loss_l1.detach().item(), global_step=iteration
                )

            # 当深度启用时 对于深度损失的处理
            # Loss on the rendered and gt depth maps.
            if self.depth_supervision:
                depth_loss = masked_depth_mse(depth_map, depth)
                depth_loss = depth_loss * self.cfg.model.depth_loss_weight
                total_loss = total_loss + depth_loss
                summary_writer.add_scalar(
                    "train/depth_loss",
                    depth_loss.detach().item(),
                    global_step=iteration,
                )

            # Dist loss from Mip360 paper.
            if self.cfg.model.dist_loss and self.cfg.model.dist_loss_weight > 0:
                svals = (weights - model.near_far[0]) / (
                    model.near_far[1] - model.near_far[0]
                )
                dist_loss = (
                    compute_dist_loss(alphas_map[..., :-1], svals)
                    * self.cfg.model.dist_weight
                )
                total_loss += dist_loss
                summary_writer.add_scalar(
                    "train/dist_loss", dist_loss.detach().item(), global_step=iteration
                )

            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

            loss = loss.detach().item()
            PSNRs.append(-10.0 * np.log(loss) / np.log(10.0))
            summary_writer.add_scalar("train/PSNR", PSNRs[-1], global_step=iteration)
            summary_writer.add_scalar("train/mse", loss, global_step=iteration)

            # Print the current values of the losses.
            if iteration % self.cfg.systems.progress_refresh_rate == 0:
                pbar.set_description(
                    f"Iteration {iteration:05d}:"
                    + f" train_psnr = {float(np.mean(PSNRs)):.2f}"
                    + f" test_psnr = {float(np.mean(PSNRs_test)):.2f}"
                    + f" mse = {loss:.6f}"
                )
                PSNRs = []

            # Decay the learning rate.
            for param_group in optimizer.param_groups:
                param_group["lr"] = param_group["lr_org"] * lr_factor

            # Evaluation for every self.cfg.systems.vis_every steps.
            if (
                iteration % self.cfg.systems.vis_every == self.cfg.systems.vis_every - 1
                and self.cfg.data.N_vis != 0
            ):
                PSNRs_test = evaluation(
                    test_dataset,
                    model,
                    self.cfg,
                    f"{self.logfolder}/imgs_vis/",
                    prefix=f"{iteration:06d}_",
                    white_bg=white_bg,
                    N_samples=nSamples,
                    ndc_ray=ndc_ray,
                    device=self.device,
                    compute_extra_metrics=False,
                )
                summary_writer.add_scalar(
                    "test/psnr", np.mean(PSNRs_test), global_step=iteration
                )

                torch.cuda.synchronize()

            # Calculate the emptiness voxel.
            if iteration in self.cfg.model.update_emptymask_list:
                if (
                    reso_cur[0] * reso_cur[1] * reso_cur[2] < 256**3
                ):  # update volume resolution
                    reso_mask = reso_cur
                model.updateEmptyMask(tuple(reso_mask))

            # Upsample the volume grid.
            if iteration in self.cfg.model.upsample_list:
                if self.cfg.model.upsampling_type == "aligned":
                    reso_cur = [reso_cur[i] * 2 - 1 for i in range(len(reso_cur))]
                else:
                    N_voxel = self.N_voxel_list.pop(0)
                    reso_cur = N_to_reso(
                        N_voxel, model.aabb, self.cfg.model.nonsquare_voxel
                    )
                time_grid = self.Time_grid_list.pop(0)
                nSamples = min(
                    self.cfg.model.nSamples,
                    cal_n_samples(reso_cur, self.cfg.model.step_ratio),
                )
                model.upsample_volume_grid(reso_cur, time_grid)

                grad_vars = model.get_optparam_groups(self.cfg.optim, 1.0)
                optimizer = torch.optim.Adam(
                    grad_vars, betas=(self.cfg.optim.beta1, self.cfg.optim.beta2)
                )

        # 训练统计：包含初始化、循环内评估、掩码更新和上采样；不含最终渲染。
        if uses_cuda:
            torch.cuda.synchronize(training_device)
        elapsed_seconds = time.perf_counter() - start_time
        peak_allocated_gib = (
            torch.cuda.max_memory_allocated(training_device) / 1024**3
            if uses_cuda else None
        )
        peak_reserved_gib = (
            torch.cuda.max_memory_reserved(training_device) / 1024**3
            if uses_cuda else None
        )
        training_stats = {
            "scope": "train() including initialization and in-loop evaluation; excludes final rendering",
            "iterations": self.cfg.optim.n_iters,
            "device": str(training_device),
            "elapsed_seconds": elapsed_seconds,
            "elapsed_minutes": elapsed_seconds / 60,
            "peak_allocated_gib": peak_allocated_gib,
            "peak_reserved_gib": peak_reserved_gib,
        }
        stats_path = Path(self.logfolder) / "training_stats.json"
        stats_path.parent.mkdir(parents=True, exist_ok=True)
        with stats_path.open("w", encoding="utf-8") as stats_file:
            json.dump(training_stats, stats_file, indent=2)
            stats_file.write("\n")

        # 同时写入 TensorBoard，便于和训练指标一起查看。
        stats_step = self.cfg.optim.n_iters
        summary_writer.add_scalar(
            "performance/train_time_seconds", elapsed_seconds, global_step=stats_step
        )
        if uses_cuda:
            summary_writer.add_scalar(
                "performance/peak_allocated_gib",
                peak_allocated_gib,
                global_step=stats_step,
            )
            summary_writer.add_scalar(
                "performance/peak_reserved_gib",
                peak_reserved_gib,
                global_step=stats_step,
            )
        summary_writer.flush()

        print(
            f"Training time: {elapsed_seconds:.2f} seconds "
            f"({elapsed_seconds / 60:.2f} minutes)"
        )
        if uses_cuda:
            print(f"Peak allocated GPU memory: {peak_allocated_gib:.2f} GiB")
            print(f"Peak reserved GPU memory: {peak_reserved_gib:.2f} GiB")
        print(f"Training statistics saved to {stats_path}")
