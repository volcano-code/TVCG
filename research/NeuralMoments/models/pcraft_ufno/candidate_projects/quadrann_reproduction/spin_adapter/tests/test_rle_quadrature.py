import numpy as np
import pytest

from spin_adapter.rle_quadrature import rle_gauss_quadrature
from research.NeuralMoments.models.pcraft_ufno.spin_pino.native_pcraft import (
    geometry_from_columns,
)


def make_geometry():
    columns = np.array([
        [1, 1, 1, 5],
        [1, 2, 2, 7],
        [2, 1, 0, 6],
        [3, 3, 1, 8],
    ], dtype=np.int32)

    return geometry_from_columns({
        "columns": columns,
        "origin": np.array([-1.0, -2.0, 0.5]),
        "h": np.array(0.25),
        "side": np.array(8, dtype=np.int32),
        "density": np.array(1.7),
    })


def test_positive_weights_and_total_mass():
    geometry = make_geometry()
    result = rle_gauss_quadrature(geometry, order=2)

    assert np.isfinite(result["weights"]).all()
    assert (result["weights"] > 0).all()

    assert result["weights"].sum() == pytest.approx(
        geometry["mass"].sum(),
        rel=1e-12,
    )


def test_each_interval_mass_is_preserved():
    geometry = make_geometry()
    result = rle_gauss_quadrature(geometry, order=2)

    recovered = np.bincount(
        result["interval_ids"],
        weights=result["weights"],
    )

    assert np.allclose(
        recovered,
        geometry["mass"],
        rtol=1e-12,
        atol=1e-14,
    )


def test_all_ten_moments_match_exact_rle_integral():
    geometry = make_geometry()
    result = rle_gauss_quadrature(geometry, order=2)

    exact = (
        geometry["mass"][:, None]
        * geometry["moment_basis"]
    ).sum(axis=0)

    assert np.allclose(
        result["moments"],
        exact,
        rtol=1e-11,
        atol=1e-12,
    )


def test_nodes_stay_inside_original_rle_intervals():
    geometry = make_geometry()
    result = rle_gauss_quadrature(geometry, order=2)

    cols = geometry["columns"]
    origin = geometry["origin"]
    h = geometry["h"]

    for i, row in enumerate(cols):
        mask = result["interval_ids"] == i
        points = result["xyz"][mask]

        x, y, z0, z1 = row

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

        assert np.all(points > lo)
        assert np.all(points < hi)


@pytest.mark.parametrize("order", [2, 3, 4])
def test_higher_order_rules_preserve_ten_moments(order):
    geometry = make_geometry()
    result = rle_gauss_quadrature(geometry, order=order)

    exact = (
        geometry["mass"][:, None]
        * geometry["moment_basis"]
    ).sum(axis=0)

    assert np.allclose(
        result["moments"],
        exact,
        rtol=1e-11,
        atol=1e-12,
    )
