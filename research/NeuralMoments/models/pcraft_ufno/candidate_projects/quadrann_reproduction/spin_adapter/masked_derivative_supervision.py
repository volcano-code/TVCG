"""Masked full and directional Jacobian supervision for frozen teachers.

The full Jacobian is differentiated with respect to the ten raw coefficients.
Restricted labels supervise ``J @ B`` in the teacher's supplied basis; the
basis need not be orthogonal or normalized. Padding is never used in arithmetic.
"""
from __future__ import annotations

import torch
from torch import Tensor


def _real_tensor(value, *, name: str, prediction: Tensor) -> Tensor:
    try:
        tensor = torch.as_tensor(value, device=prediction.device)
    except (TypeError, ValueError, RuntimeError) as exc:
        raise ValueError(f"{name} must be a real numeric tensor") from exc
    if tensor.is_complex() or tensor.dtype == torch.bool:
        raise ValueError(f"{name} must be real numeric data, not complex or boolean")
    # Convert Python scalars directly to the prediction dtype: converting
    # through the default float32 dtype would round double-precision scales.
    return torch.as_tensor(value, device=prediction.device, dtype=prediction.dtype).detach()


def _mask(value, *, name: str, rows: int, default: bool, device) -> Tensor:
    if value is None:
        return torch.full((rows,), default, dtype=torch.bool, device=device)
    try:
        mask = torch.as_tensor(value, device=device)
    except (TypeError, ValueError, RuntimeError) as exc:
        raise ValueError(f"{name} must be a boolean mask with shape [Q]") from exc
    if mask.dtype != torch.bool or mask.shape != (rows,):
        raise ValueError(f"{name} must be a boolean mask with shape [Q]")
    return mask.detach()


def _require_finite(value: Tensor, name: str) -> None:
    if not torch.isfinite(value).all():
        raise ValueError(f"active {name} entries must be finite")


def _scale(value, *, name: str, prediction: Tensor, restricted: bool,
           active: bool) -> Tensor:
    scale = _real_tensor(value, name=name, prediction=prediction)
    shapes = ((), (10,), (10, 1)) if restricted else ((), (10, 10))
    if tuple(scale.shape) not in shapes:
        allowed = "scalar, [10], or [10,1]" if restricted else "scalar or [10,10]"
        raise ValueError(f"{name} must have shape {allowed}")
    if active:
        _require_finite(scale, name)
        if (scale <= 0).any():
            raise ValueError(f"active {name} entries must be positive")
    if restricted and scale.shape == (10,):
        scale = scale[:, None]
    return scale


def derivative_supervision_loss(
    prediction: Tensor,
    full_target: Tensor | None,
    *,
    derivative_valid=None,
    restricted_basis=None,
    restricted_target=None,
    restricted_valid=None,
    restricted_dimension=None,
    jacobian_scale=1.,
    restricted_scale=1.,
) -> dict[str, Tensor]:
    """Average derivative errors over supervised queries, with equal row weight.

    ``prediction`` and an optional full target have shape ``[Q,10,10]``.
    Restricted basis and target have matching padded shape ``[Q,10,D]``;
    ``restricted_dimension`` supplies each row's active prefix length. When
    omitted, all ``D`` columns are used. Finite exact integer dimensions in
    ``[1,D]`` are required only for active restricted rows; inactive dimension
    values may contain NaN padding.

    Masks must be boolean ``[Q]`` tensors. An omitted mask selects all rows
    when that target is supplied, and no rows otherwise. Full and restricted
    masks must be disjoint. Teacher targets, bases, and scales are detached.
    Only selected rows and active basis/target columns enter arithmetic or
    finite checks, so excluded rows and padded columns may contain NaNs.

    Each full row averages its 100 squared normalized errors. Each restricted
    row averages its ``10*d`` squared errors after projecting the raw prediction
    as ``J @ B``. ``restricted_scale`` is an independent scalar or per-output
    scale (``[10]`` or ``[10,1]``), never an elementwise full-Jacobian scale
    projected into the restricted basis. ``jacobian_scale`` is scalar or
    ``[10,10]``. Scales used by an active branch must be finite and positive.

    ``loss`` is the average of the supervised row means. ``full_loss`` and
    ``restricted_loss`` are the respective branch means. Counts and
    ``restricted_dimensions`` (the sum of active directional dimensions) are
    scalar int64 tensors. Empty branches return detached scalar zeros; when
    all derivative masks are off, this function builds no derivative graph.
    """
    if not isinstance(prediction, Tensor) or not prediction.is_floating_point():
        raise ValueError("prediction must be a real floating-point tensor with shape [Q,10,10]")
    if prediction.ndim != 3 or prediction.shape[1:] != (10, 10):
        raise ValueError("prediction must have shape [Q,10,10]")
    rows = prediction.shape[0]
    full_mask = _mask(derivative_valid, name="derivative_valid", rows=rows,
                      default=full_target is not None, device=prediction.device)
    has_restricted = restricted_basis is not None or restricted_target is not None
    restricted_mask = _mask(restricted_valid, name="restricted_valid", rows=rows,
                            default=has_restricted, device=prediction.device)
    if (full_mask & restricted_mask).any():
        raise ValueError("full and restricted derivative masks must be disjoint")

    full_count = full_mask.sum(dtype=torch.int64)
    restricted_count = restricted_mask.sum(dtype=torch.int64)
    full_active = bool(full_count)
    restricted_active = bool(restricted_count)
    if full_active and full_target is None:
        raise ValueError("full_target is required for active derivative_valid rows")
    full = None
    if full_target is not None:
        full = _real_tensor(full_target, name="full_target", prediction=prediction)
        if full.shape != prediction.shape:
            raise ValueError("full_target must have shape [Q,10,10]")

    basis = target = None
    padded_dimension = 0
    if has_restricted:
        if restricted_basis is None or restricted_target is None:
            raise ValueError("restricted_basis and restricted_target must be provided together")
        basis = _real_tensor(restricted_basis, name="restricted_basis", prediction=prediction)
        target = _real_tensor(restricted_target, name="restricted_target", prediction=prediction)
        if (basis.ndim != 3 or basis.shape[:2] != (rows, 10)
                or basis.shape[2] < 1 or target.shape != basis.shape):
            raise ValueError("restricted_basis and restricted_target must have matching shape [Q,10,D], D>=1")
        padded_dimension = basis.shape[2]
    elif restricted_active:
        raise ValueError("restricted_basis and restricted_target are required for active restricted_valid rows")

    if restricted_dimension is None:
        dimensions = torch.full((rows,), padded_dimension, dtype=torch.int64,
                                device=prediction.device)
    else:
        try:
            dimensions = torch.as_tensor(restricted_dimension, device=prediction.device).detach()
        except (TypeError, ValueError, RuntimeError) as exc:
            raise ValueError("restricted_dimension must be a real vector with shape [Q]") from exc
        if dimensions.is_complex() or dimensions.dtype == torch.bool or dimensions.shape != (rows,):
            raise ValueError("restricted_dimension must be a real vector with shape [Q]")
    active_dimensions = dimensions[restricted_mask].to(dtype=torch.float64)
    if restricted_active:
        _require_finite(active_dimensions, "restricted_dimension")
        if ((active_dimensions < 1) | (active_dimensions > padded_dimension)
                | (active_dimensions != active_dimensions.round())).any():
            raise ValueError("active restricted_dimension values must be integers in [1,D]")
    active_dimensions = active_dimensions.to(dtype=torch.int64)

    full_scale = _scale(jacobian_scale, name="jacobian_scale", prediction=prediction,
                        restricted=False, active=full_active)
    directional_scale = _scale(restricted_scale, name="restricted_scale", prediction=prediction,
                               restricted=True, active=restricted_active)
    zero = prediction.new_zeros(())
    full_rows = prediction.new_empty((0,))
    restricted_rows = prediction.new_empty((0,))
    if full_active:
        selected_prediction = prediction[full_mask]
        selected_target = full[full_mask]
        _require_finite(selected_prediction, "full prediction")
        _require_finite(selected_target, "full_target")
        full_rows = ((selected_prediction - selected_target) / full_scale).square().mean(dim=(1, 2))

    if restricted_active:
        selected_indices = restricted_mask.nonzero(as_tuple=False).flatten()
        row_losses = []
        # Group by d, selecting the valid column prefix before projection.
        for dimension in active_dimensions.unique(sorted=True).tolist():
            indices = selected_indices[active_dimensions == dimension]
            selected_prediction = prediction.index_select(0, indices)
            selected_basis = basis.index_select(0, indices)[:, :, :dimension]
            selected_target = target.index_select(0, indices)[:, :, :dimension]
            _require_finite(selected_prediction, "restricted prediction")
            _require_finite(selected_basis, "restricted_basis")
            _require_finite(selected_target, "restricted_target")
            error = (selected_prediction @ selected_basis - selected_target) / directional_scale
            row_losses.append(error.square().mean(dim=(1, 2)))
        restricted_rows = torch.cat(row_losses)

    count = full_count + restricted_count
    loss = (full_rows.sum() + restricted_rows.sum()) / count if bool(count) else zero
    if not torch.isfinite(loss):
        raise FloatingPointError("derivative supervision produced a nonfinite loss")
    return {
        "loss": loss,
        "full_loss": full_rows.mean() if full_active else zero,
        "restricted_loss": restricted_rows.mean() if restricted_active else zero,
        "full_count": full_count,
        "restricted_count": restricted_count,
        "restricted_dimensions": active_dimensions.sum(dtype=torch.int64),
    }
