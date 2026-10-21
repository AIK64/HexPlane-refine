"""Shared depth rendering, supervision and evaluation conventions."""

import torch


def render_depth(weights, z_vals, far, mode="far"):
    """Return depth in ray-parameter units, not camera-Z or NDC metric depth."""
    accumulated = weights.sum(dim=-1)
    depth = (weights * z_vals).sum(dim=-1)
    if mode == "far":
        return depth + (1.0 - accumulated).clamp_min(0) * far
    if mode == "conditional":
        return torch.where(
            accumulated > 1e-8,
            depth / accumulated.clamp_min(1e-8),
            torch.full_like(depth, far),
        )
    raise ValueError(f"Unknown depth_render_mode: {mode}")


def depth_pairs(prediction, target):
    """Flatten matching per-pixel depths and mask invalid reference values."""
    if target is None:
        raise ValueError("Depth supervision requires reference depths from the dataset.")
    if prediction.numel() != target.numel():
        raise ValueError(
            f"Depth size mismatch: prediction {tuple(prediction.shape)}, "
            f"reference {tuple(target.shape)}"
        )
    prediction = prediction.reshape(-1)
    target = target.to(device=prediction.device, dtype=prediction.dtype).reshape(-1)
    valid = torch.isfinite(target) & (target > 0)
    # Do not silently hide numerical failures in predictions at supervised pixels.
    if not torch.isfinite(prediction[valid]).all():
        raise ValueError("Non-finite predicted depth at a valid reference pixel.")
    return prediction, target, valid


def masked_depth_mse(prediction, target):
    prediction, target, valid = depth_pairs(prediction, target)
    if not valid.any():
        # Empty batches contribute zero without breaking the autograd graph.
        return prediction[:0].sum()
    return (prediction[valid] - target[valid]).square().mean()


def normalized_depth_pairs(prediction, target):
    """Normalize one image's valid camera-Z depths; never mix independent images."""
    prediction, target, valid = depth_pairs(prediction, target)
    if valid.sum() < 8:
        return None  # Too few foreground pixels to estimate a stable image statistic.
    prediction, target = prediction[valid], target[valid]
    target = target - target.median()
    target_scale = target.abs().mean()
    if target_scale <= 1e-6:
        return None  # Constant pseudo-depth carries no relative shape information.
    prediction = prediction - prediction.median()
    prediction = prediction / prediction.abs().mean().clamp_min(1e-6)
    return prediction, target / target_scale


def relative_depth_l1(prediction, target):
    normalized = normalized_depth_pairs(prediction, target)
    if normalized is None:
        return prediction.reshape(-1)[:0].sum()
    return (normalized[0] - normalized[1]).abs().mean()
