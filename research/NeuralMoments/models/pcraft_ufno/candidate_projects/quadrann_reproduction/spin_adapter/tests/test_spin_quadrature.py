import numpy as np
import pytest
import torch

from spin_adapter.rle_quadrature import rle_gauss_quadrature
from spin_adapter.spin_quadrature import (
    apply_interval_mass_weights,
    integrate_ten_moments,
)
from research.NeuralMoments.models.pcraft_ufno.spin_pino.native_pcraft import (
    geometry_from_columns,
)


def geometry():
    columns = np.array([
        [1, 1, 1, 5],
        [1, 2, 2, 7],
        [2, 1, 0, 6],
        [3, 3, 1, 8],
        [4, 3, 0, 7],
    ], dtype=np.int32)

    return geometry_from_columns({
        "columns": columns,
        "origin": np.array([-1., -2., .5]),
        "h": np.array(.25),
        "side": np.array(8, dtype=np.int32),
        "density": np.array(1.7),
    })


def test_zero_logits_recover_exact_original_rle_quadrature():
    g = geometry()
    base = rle_gauss_quadrature(g, order=2)

    # Softmax(log(m)) * total_mass == original interval masses.
    logits = torch.zeros(len(g["mass"]), dtype=torch.float64)
    prior = torch.as_tensor(g["mass"], dtype=torch.float64)

    learned = (
        torch.softmax(logits + prior.log(), dim=0)
        * prior.sum()
    ).numpy()

    result = apply_interval_mass_weights(base, learned)

    assert np.allclose(
        result["weights"], base["weights"],
        rtol=1e-12, atol=1e-13,
    )

    assert np.allclose(
        integrate_ten_moments(result),
        g["original_moments"],
        rtol=1e-11, atol=1e-12,
    )


def test_nonzero_logits_preserve_total_and_interval_mass():
    g = geometry()
    base = rle_gauss_quadrature(g, order=2)

    logits = np.linspace(-2., 2., len(g["mass"]))
    mass = g["mass"]

    shifted = logits + np.log(mass)
    shifted -= shifted.max()

    probabilities = np.exp(shifted)
    probabilities /= probabilities.sum()

    learned = probabilities * mass.sum()

    result = apply_interval_mass_weights(base, learned)

    assert np.all(result["weights"] > 0)
    assert result["weights"].sum() == pytest.approx(
        mass.sum(), rel=1e-12
    )

    recovered = np.bincount(
        result["interval_ids"],
        weights=result["weights"],
        minlength=len(learned),
    )

    assert np.allclose(
        recovered, learned,
        rtol=1e-11, atol=1e-13,
    )


def test_chunked_and_full_moment_integrals_match():
    g = geometry()
    base = rle_gauss_quadrature(g, order=3)

    full = integrate_ten_moments(
        base, chunk_nodes=len(base["weights"])
    )

    chunked = integrate_ten_moments(
        base, chunk_nodes=7
    )

    assert np.allclose(
        full, chunked,
        rtol=1e-12, atol=1e-12,
    )


def test_bad_mass_input_is_rejected():
    g = geometry()
    base = rle_gauss_quadrature(g, order=2)

    bad = g["mass"].copy()
    bad[0] = -1

    with pytest.raises(ValueError):
        apply_interval_mass_weights(base, bad)


def test_wrong_total_mass_is_rejected():
    g = geometry()
    base = rle_gauss_quadrature(g, order=2)

    bad = g["mass"] * 2

    with pytest.raises(ValueError):
        apply_interval_mass_weights(base, bad)
