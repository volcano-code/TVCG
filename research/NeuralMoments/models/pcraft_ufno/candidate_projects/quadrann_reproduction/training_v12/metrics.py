"""Raw-unit and normalized moment evaluation metrics."""

import numpy as np


def evaluate_moments(prediction, target, scales):
    prediction = np.asarray(prediction, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    scales = np.asarray(scales, dtype=np.float64)

    if prediction.shape != target.shape:
        raise ValueError("prediction/target shape mismatch")

    if prediction.ndim != 2 or prediction.shape[1] != 10:
        raise ValueError("expected [N,10] moments")

    error = prediction - target
    normalized = error / scales[None, :]

    per_sample_relative = (
        np.linalg.norm(error, axis=1)
        / np.maximum(np.linalg.norm(target, axis=1), 1e-12)
    )

    return {
        "count": int(len(target)),
        "mae_by_moment": np.abs(error).mean(axis=0).tolist(),
        "rmse_by_moment": np.sqrt(
            np.square(error).mean(axis=0)
        ).tolist(),
        "normalized_rmse": float(
            np.sqrt(np.square(normalized).mean())
        ),
        "relative_l2_mean": float(
            per_sample_relative.mean()
        ),
        "relative_l2_p95": float(
            np.percentile(per_sample_relative, 95)
        ),
        "max_absolute_error": float(
            np.abs(error).max()
        ),
    }
