"""RLE-aware positive Gaussian quadrature for Spin geometry.

Each native RLE interval is integrated independently using a tensor-product
Gauss-Legendre rule. No 512^3 dense voxel expansion is performed.

Moment order:
(1, x, y, z, x^2, y^2, z^2, yz, xz, xy)
"""
from __future__ import annotations

import numpy as np


def native_moment_basis(xyz: np.ndarray) -> np.ndarray:
    x, y, z = np.asarray(xyz, dtype=np.float64).T
    return np.column_stack((
        np.ones_like(x),
        x, y, z,
        x*x, y*y, z*z,
        y*z, x*z, x*y,
    ))


def rle_gauss_quadrature(
    geometry: dict,
    order: int = 2,
) -> dict:
    """Generate positive physical quadrature nodes for native RLE intervals.

    geometry must come from geometry_from_columns().
    Returns physical coordinates and mass-weighted integration weights.
    """
    if not isinstance(order, int) or order < 2:
        raise ValueError("order must be an integer >= 2")

    columns = np.asarray(geometry["columns"], dtype=np.int64)
    origin = np.asarray(geometry["origin"], dtype=np.float64)
    h = float(geometry["h"])
    density = float(geometry["density"])

    if columns.ndim != 2 or columns.shape[1] != 4 or len(columns) == 0:
        raise ValueError("columns must be nonempty integer [N,4]")
    if origin.shape != (3,) or not np.isfinite(origin).all():
        raise ValueError("origin must be finite [3]")
    if not np.isfinite(h) or h <= 0 or not np.isfinite(density) or density <= 0:
        raise ValueError("h and density must be finite and positive")
    if (columns[:, 3] <= columns[:, 2]).any():
        raise ValueError("RLE intervals must have positive z extent")

    nodes_1d, weights_1d = np.polynomial.legendre.leggauss(order)

    # Tensor-product reference nodes and strictly positive weights.
    ijk = np.stack(
        np.meshgrid(
            np.arange(order),
            np.arange(order),
            np.arange(order),
            indexing="ij",
        ),
        axis=-1,
    ).reshape(-1, 3)

    reference_nodes = nodes_1d[ijk]
    reference_weights = np.prod(weights_1d[ijk], axis=-1)

    n_intervals = len(columns)
    n_local = order ** 3

    xyz = np.empty((n_intervals, n_local, 3), dtype=np.float64)
    mass_weights = np.empty((n_intervals, n_local), dtype=np.float64)

    interval_mass = np.asarray(geometry["mass"], dtype=np.float64)

    if interval_mass.shape != (n_intervals,):
        raise ValueError("geometry mass does not match RLE intervals")

    for i, (x, y, z0, z1) in enumerate(columns):
        lo = np.array([
            origin[0] + x*h,
            origin[1] + y*h,
            origin[2] + z0*h,
        ])
        hi = np.array([
            origin[0] + (x+1)*h,
            origin[1] + (y+1)*h,
            origin[2] + z1*h,
        ])

        center = (lo + hi) * 0.5
        half = (hi - lo) * 0.5

        xyz[i] = center + reference_nodes * half

        jacobian = float(np.prod(half))
        mass_weights[i] = (
            density * jacobian * reference_weights
        )

    xyz = xyz.reshape(-1, 3)
    mass_weights = mass_weights.reshape(-1)

    interval_centers = np.column_stack((
        origin[0] + (columns[:, 0] + 0.5) * h,
        origin[1] + (columns[:, 1] + 0.5) * h,
        origin[2] + (
            columns[:, 2] + columns[:, 3]
        ) * 0.5 * h,
    ))

    interval_ids = np.repeat(
        np.arange(n_intervals, dtype=np.int64),
        n_local,
    )

    if not np.isfinite(xyz).all():
        raise ValueError("nonfinite quadrature nodes")
    if not np.isfinite(mass_weights).all() or (mass_weights <= 0).any():
        raise ValueError("quadrature weights must be finite and positive")

    recovered_mass = np.bincount(
        interval_ids,
        weights=mass_weights,
        minlength=n_intervals,
    )

    if not np.allclose(
        recovered_mass, interval_mass,
        rtol=1e-12, atol=1e-14,
    ):
        raise RuntimeError("quadrature does not preserve interval mass")

    basis = native_moment_basis(xyz)

    moments = np.einsum(
        "n,ni->i", mass_weights, basis
    )

    exact_moments = (
        interval_mass[:, None]
        * np.asarray(geometry["moment_basis"], dtype=np.float64)
    ).sum(axis=0)

    if not np.allclose(
        moments, exact_moments,
        rtol=1e-11, atol=1e-12,
    ):
        raise RuntimeError("quadrature failed exact ten-moment audit")

    return {
        "xyz": xyz,
        "interval_centers": interval_centers,
        "weights": mass_weights,
        "interval_ids": interval_ids,
        "interval_mass": interval_mass.copy(),
        "moment_basis": basis,
        "moments": moments,
        "n_intervals": n_intervals,
        "n_nodes": len(xyz),
        "order": order,
        "geometry_sha256": geometry.get("geometry_sha256"),
    }
