import math

import cv2
import numpy as np


def visualize_depth_numpy(depth, minmax=None, cmap=cv2.COLORMAP_JET):
    """
    depth: (H, W)
    """

    x = np.nan_to_num(depth, nan=0.0, posinf=0.0, neginf=0.0)
    if minmax is None:
        positive = x[x > 0]
        mi = positive.min() if positive.size else 0.0
        ma = positive.max() if positive.size else 1.0
    else:
        mi, ma = minmax

    x = (x - mi) / (ma - mi + 1e-8)  # normalize to 0~1
    x = (255 * np.clip(x, 0, 1)).astype(np.uint8)
    x_ = cv2.applyColorMap(x, cmap)
    return x_, [mi, ma]


def N_to_reso(n_voxels, bbox, adjusted_grid=True):
    if adjusted_grid:
        xyz_min, xyz_max = bbox
        voxel_size = ((xyz_max - xyz_min).prod() / n_voxels).pow(1 / 3)
        return ((xyz_max - xyz_min) / voxel_size).long().tolist()
    else:
        # grid_each = n_voxels.pow(1 / 3)
        grid_each = math.pow(n_voxels, 1 / 3)
        return [int(grid_each), int(grid_each), int(grid_each)]
