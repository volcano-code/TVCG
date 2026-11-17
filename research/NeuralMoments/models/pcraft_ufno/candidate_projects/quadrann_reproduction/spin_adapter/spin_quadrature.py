"""Mass-preserving RLE interval quadrature for Spin-QuadrANN.

Upstream-inspired:
- geometry-conditioned positive quadrature weights.

Spin-specific:
- native RLE interval mass prior;
- exact zero-logit recovery of the original RLE quadrature;
- interval-level mass conservation;
- chunked ten-moment integration.
"""
from __future__ import annotations

import numpy as np


def apply_interval_mass_weights(
    quadrature: dict,
    learned_interval_mass: np.ndarray,
    chunk_nodes: int = 65536,
) -> dict:
    """Redistribute learned interval mass over existing Gauss nodes.

    The relative Gauss weights inside each RLE interval are preserved.
    """
    base_weights = np.asarray(
        quadrature["weights"], dtype=np.float64
    )
    interval_ids = np.asarray(
        quadrature["interval_ids"], dtype=np.int64
    )
    interval_mass = np.asarray(
        quadrature["interval_mass"], dtype=np.float64
    )
    learned = np.asarray(
        learned_interval_mass, dtype=np.float64
    )
    xyz = np.asarray(quadrature["xyz"], dtype=np.float64)

    n_intervals = len(interval_mass)
    n_nodes = len(base_weights)

    if xyz.shape != (n_nodes, 3):
        raise ValueError("invalid quadrature xyz")
    if interval_ids.shape != (n_nodes,):
        raise ValueError("invalid interval_ids")
    if learned.shape != (n_intervals,):
        raise ValueError("learned interval mass must be [N]")
    if not np.isfinite(base_weights).all() or (base_weights <= 0).any():
        raise ValueError("base quadrature weights must be positive")
    if not np.isfinite(learned).all() or (learned <= 0).any():
        raise ValueError("learned interval masses must be positive")
    if interval_ids.min() < 0 or interval_ids.max() >= n_intervals:
        raise ValueError("interval id out of bounds")
    if chunk_nodes < 1:
        raise ValueError("chunk_nodes must be positive")

    base_interval_mass = np.bincount(
        interval_ids,
        weights=base_weights,
        minlength=n_intervals,
    )

    if not np.allclose(
        base_interval_mass,
        interval_mass,
        rtol=1e-11,
        atol=1e-13,
    ):
        raise ValueError("base interval quadrature mass mismatch")

    total_mass = float(interval_mass.sum())
    learned_total = float(learned.sum())

    if not np.isclose(
        learned_total, total_mass,
        rtol=1e-10,
        atol=1e-12,
    ):
        raise ValueError("learned interval masses do not conserve total mass")

    # Preserve each interval's local Gauss distribution.
    local_fraction = (
        base_weights / base_interval_mass[interval_ids]
    )
    node_weights = learned[interval_ids] * local_fraction

    if not np.isfinite(node_weights).all() or (node_weights <= 0).any():
        raise RuntimeError("invalid learned node weights")

    recovered = np.bincount(
        interval_ids,
        weights=node_weights,
        minlength=n_intervals,
    )

    if not np.allclose(
        recovered, learned,
        rtol=1e-11,
        atol=1e-13,
    ):
        raise RuntimeError("interval mass redistribution failed")

    return {
        "xyz": xyz,
        "weights": node_weights,
        "interval_ids": interval_ids,
        "interval_mass": learned.copy(),
        "total_mass": float(node_weights.sum()),
        "n_nodes": n_nodes,
        "n_intervals": n_intervals,
        "chunk_nodes": chunk_nodes,
    }


def integrate_ten_moments(
    quadrature: dict,
    chunk_nodes: int = 65536,
) -> np.ndarray:
    """Chunked integration of the fixed ten-component moment basis."""
    xyz = np.asarray(quadrature["xyz"], dtype=np.float64)
    weights = np.asarray(quadrature["weights"], dtype=np.float64)

    if xyz.ndim != 2 or xyz.shape[1] != 3:
        raise ValueError("xyz must be [N,3]")
    if weights.shape != (len(xyz),):
        raise ValueError("weights must be [N]")
    if not np.isfinite(xyz).all() or not np.isfinite(weights).all():
        raise ValueError("nonfinite integration input")
    if (weights <= 0).any():
        raise ValueError("weights must be positive")
    if chunk_nodes < 1:
        raise ValueError("chunk_nodes must be positive")

    result = np.zeros(10, dtype=np.float64)

    for start in range(0, len(xyz), chunk_nodes):
        stop = min(start + chunk_nodes, len(xyz))

        x = xyz[start:stop, 0]
        y = xyz[start:stop, 1]
        z = xyz[start:stop, 2]
        w = weights[start:stop]

        basis = np.column_stack((
            np.ones_like(x),
            x, y, z,
            x*x, y*y, z*z,
            y*z, x*z, x*y,
        ))

        result += np.einsum(
            "n,ni->i", w, basis
        )

    return result
