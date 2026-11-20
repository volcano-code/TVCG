"""Independent reader and verifier for Spin-It initial material-state v1 exports."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np


SCHEMA = "spin-it-2014-initial-material-state-v1"


@dataclass(frozen=True)
class VerifiedInitialMaterialState:
    domain_artifact_id: str
    grid_dimensions: tuple[int, int, int]
    source_material_mask: np.ndarray
    fixed_material_mask: np.ndarray
    design_domain_mask: np.ndarray
    moments_model: np.ndarray
    moments_target: np.ndarray
    si_scale_registered: bool


def _moments_for_mask(
    mask: np.ndarray,
    dimensions: np.ndarray,
    minimum: np.ndarray,
    maximum: np.ndarray,
    linear: np.ndarray | None = None,
    translation: np.ndarray | None = None,
) -> np.ndarray:
    spacing = (maximum - minimum) / dimensions
    result = np.zeros(10, dtype=np.float64)
    for flat in np.flatnonzero(mask):
        x = int(flat % dimensions[0])
        y = int((flat // dimensions[0]) % dimensions[1])
        z = int(flat // (dimensions[0] * dimensions[1]))
        center = minimum + (np.array([x, y, z], dtype=np.float64) + 0.5) * spacing
        if linear is not None:
            center = linear @ center + translation
            cell_linear = linear @ np.diag(spacing)
            covariance = cell_linear @ cell_linear.T / 12.0
            cell_volume = abs(float(np.linalg.det(cell_linear)))
        else:
            covariance = np.diag(spacing * spacing / 12.0)
            cell_volume = float(np.prod(spacing))
        result[0] += cell_volume
        result[1:4] += cell_volume * center
        second = cell_volume * (np.outer(center, center) + covariance)
        result[4:7] += np.diag(second)
        result[7] += second[1, 2]
        result[8] += second[0, 2]
        result[9] += second[0, 1]
    return result


def validate_initial_material_state(
    diagnostics: Mapping[str, Any],
    *,
    rtol: float = 1e-10,
    atol: float = 1e-12,
) -> VerifiedInitialMaterialState:
    state = diagnostics.get("initial_state")
    if not isinstance(state, Mapping) or state.get("schema") != SCHEMA:
        raise ValueError(f"missing or unsupported initial_state schema: {SCHEMA}")
    if state.get("scope") != "INITIAL_SOURCE_DOMAIN_NOT_FINAL_OPTIMIZED_MATERIAL":
        raise ValueError("initial_state scope mismatch")
    dims = np.asarray(state.get("grid_dimensions"), dtype=np.int64)
    minimum = np.asarray(state.get("grid_minimum_model"), dtype=np.float64)
    maximum = np.asarray(state.get("grid_maximum_model"), dtype=np.float64)
    if dims.shape != (3,) or np.any(dims <= 0):
        raise ValueError("grid_dimensions must be positive [3]")
    if minimum.shape != (3,) or maximum.shape != (3,) or not np.isfinite(minimum).all() or not np.isfinite(maximum).all() or np.any(maximum <= minimum):
        raise ValueError("invalid grid bounds")
    count = int(np.prod(dims))
    masks = {}
    for key in ("source_material_mask", "source_carveable_mask", "fixed_material_mask", "design_domain_mask"):
        value = np.asarray(state.get(key), dtype=np.uint8)
        if value.shape != (count,) or np.any(value > 1):
            raise ValueError(f"{key} must be a binary x-fast mask of length {count}")
        masks[key] = value.astype(bool)
    source = masks["source_material_mask"]
    carveable = masks["source_carveable_mask"]
    fixed = masks["fixed_material_mask"]
    design = masks["design_domain_mask"]
    if np.any(carveable & ~source):
        raise ValueError("carveable mask contains cells outside source material")
    if not np.array_equal(fixed, source & ~carveable):
        raise ValueError("fixed_material_mask does not equal source_material & ~carveable")
    if not np.array_equal(design, source & carveable):
        raise ValueError("design_domain_mask does not equal source_material & carveable")
    if state.get("cell_order") != "x + nx*(y + ny*z)":
        raise ValueError("unsupported cell ordering")
    spacing = (maximum - minimum) / dims
    reported_spacing = np.asarray(state.get("grid_spacing_model"), dtype=np.float64)
    if reported_spacing.shape != (3,) or not np.allclose(spacing, reported_spacing, rtol=rtol, atol=atol):
        raise ValueError("reported grid spacing disagrees with bounds/dimensions")

    linear = np.asarray(state.get("model_to_target_linear"), dtype=np.float64)
    translation = np.asarray(state.get("model_to_target_translation"), dtype=np.float64)
    if linear.shape != (3, 3) or translation.shape != (3,) or not np.isfinite(linear).all() or not np.isfinite(translation).all():
        raise ValueError("invalid model-to-target transform")
    if not np.allclose(linear.T @ linear, np.eye(3), rtol=rtol, atol=atol):
        raise ValueError("model-to-target linear transform must be orthonormal")
    if not np.isclose(abs(np.linalg.det(linear)), 1.0, rtol=rtol, atol=atol):
        raise ValueError("model-to-target transform must preserve volume")

    expected_model = _moments_for_mask(source, dims, minimum, maximum)
    expected_target = _moments_for_mask(source, dims, minimum, maximum, linear, translation)
    supplied_model = np.asarray(state.get("initial_geometric_moments_unit_density_model"), dtype=np.float64)
    supplied_target = np.asarray(state.get("initial_geometric_moments_unit_density_target"), dtype=np.float64)
    if supplied_model.shape != (10,) or not np.allclose(supplied_model, expected_model, rtol=rtol, atol=atol):
        raise ValueError("model-frame initial geometric moments failed independent recomputation")
    if supplied_target.shape != (10,) or not np.allclose(supplied_target, expected_target, rtol=rtol, atol=atol):
        raise ValueError("target-frame initial geometric moments failed independent recomputation")
    if state.get("moment_semantics") != "geometric_volume_moments_at_unit_density_in_source_coordinate_units":
        raise ValueError("unsupported moment semantics")
    si_registered = state.get("physical_si_scale_registered")
    if not isinstance(si_registered, bool):
        raise ValueError("physical_si_scale_registered must be explicit boolean")
    return VerifiedInitialMaterialState(
        domain_artifact_id=str(state.get("domain_artifact_id", "")),
        grid_dimensions=tuple(int(x) for x in dims),
        source_material_mask=source,
        fixed_material_mask=fixed,
        design_domain_mask=design,
        moments_model=expected_model,
        moments_target=expected_target,
        si_scale_registered=si_registered,
    )


__all__ = ["SCHEMA", "VerifiedInitialMaterialState", "validate_initial_material_state"]
