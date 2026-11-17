"""Moment-only training losses. Jacobian loss is intentionally disabled."""

import torch


def moment_mse(prediction, target, scales):
    if prediction.shape != target.shape:
        raise ValueError("prediction/target shape mismatch")

    if prediction.ndim != 2 or prediction.shape[-1] != 10:
        raise ValueError("moments must have shape [B,10]")

    if scales.shape != (10,):
        raise ValueError("scales must have shape [10]")

    if not torch.isfinite(prediction).all():
        raise FloatingPointError("nonfinite prediction")

    if not torch.isfinite(target).all():
        raise FloatingPointError("nonfinite target")

    normalized_error = (
        prediction - target
    ) / scales.unsqueeze(0)

    return normalized_error.square().mean()


def moment_mae(prediction, target, scales):
    return (
        (prediction - target).abs()
        / scales.unsqueeze(0)
    ).mean()
