from .dnerf_dataset import DNerfDataset
from .neural_3D_dataset_NDC import Neural3D_NDC_Dataset


def _depth_options(cfg, split):
    depth_dir = getattr(cfg.data, f"depth_{split}_dir")
    if cfg.data.dataset_name != "dnerf" and depth_dir is not None:
        raise ValueError("External depth files currently support DNeRF only, not NDC rays.")
    return dict(
        depth_dir=depth_dir,
        depth_file_pattern=cfg.data.depth_file_pattern,
        depth_type=cfg.data.depth_type,
        depth_scale=cfg.data.depth_scale,
    )


def get_train_dataset(cfg, is_stack=False):
    depth_options = _depth_options(cfg, "train")
    if cfg.data.dataset_name == "dnerf":
        train_dataset = DNerfDataset(
            cfg.data.datadir,
            "train",
            cfg.data.downsample,
            is_stack=is_stack,
            cal_fine_bbox=cfg.data.cal_fine_bbox,
            N_vis=cfg.data.N_vis,
            time_scale=cfg.data.time_scale,
            scene_bbox_min=cfg.data.scene_bbox_min,
            scene_bbox_max=cfg.data.scene_bbox_max,
            N_random_pose=cfg.data.N_random_pose,
            **depth_options,
        )
    elif cfg.data.dataset_name == "neural3D_NDC":
        train_dataset = Neural3D_NDC_Dataset(
            cfg.data.datadir,
            "train",
            cfg.data.downsample,
            is_stack=is_stack,
            cal_fine_bbox=cfg.data.cal_fine_bbox,
            N_vis=cfg.data.N_vis,
            time_scale=cfg.data.time_scale,
            scene_bbox_min=cfg.data.scene_bbox_min,
            scene_bbox_max=cfg.data.scene_bbox_max,
            N_random_pose=cfg.data.N_random_pose,
            bd_factor=cfg.data.nv3d_ndc_bd_factor,
            eval_step=cfg.data.nv3d_ndc_eval_step,
            eval_index=cfg.data.nv3d_ndc_eval_index,
            sphere_scale=cfg.data.nv3d_ndc_sphere_scale,
        )
    else:
        raise NotImplementedError("No such dataset")
    return train_dataset


def get_test_dataset(cfg, is_stack=True):
    depth_options = _depth_options(cfg, "test")
    if cfg.data.dataset_name == "dnerf":
        test_dataset = DNerfDataset(
            cfg.data.datadir,
            "test",
            cfg.data.downsample,
            is_stack=is_stack,
            cal_fine_bbox=cfg.data.cal_fine_bbox,
            N_vis=cfg.data.N_vis,
            time_scale=cfg.data.time_scale,
            scene_bbox_min=cfg.data.scene_bbox_min,
            scene_bbox_max=cfg.data.scene_bbox_max,
            N_random_pose=cfg.data.N_random_pose,
            **depth_options,
        )
    elif cfg.data.dataset_name == "neural3D_NDC":
        test_dataset = Neural3D_NDC_Dataset(
            cfg.data.datadir,
            "test",
            cfg.data.downsample,
            is_stack=is_stack,
            cal_fine_bbox=cfg.data.cal_fine_bbox,
            N_vis=cfg.data.N_vis,
            time_scale=cfg.data.time_scale,
            scene_bbox_min=cfg.data.scene_bbox_min,
            scene_bbox_max=cfg.data.scene_bbox_max,
            N_random_pose=cfg.data.N_random_pose,
            bd_factor=cfg.data.nv3d_ndc_bd_factor,
            eval_step=cfg.data.nv3d_ndc_eval_step,
            eval_index=cfg.data.nv3d_ndc_eval_index,
            sphere_scale=cfg.data.nv3d_ndc_sphere_scale,
        )
    else:
        raise NotImplementedError("No such dataset")
    return test_dataset
