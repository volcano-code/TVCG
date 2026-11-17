import numpy as np
import torch
import pytest

from training_v12.normalization import MomentNormalizer
from training_v12.losses import moment_mse, moment_mae
from training_v12.metrics import evaluate_moments


def test_normalizer_uses_train_only():
    class FakeCorpus:
        def iter_split(self, split):
            if split != "train":
                raise AssertionError("non-train data accessed")
            for i in range(2):
                values = np.full((3, 10), i + 1.0)
                yield {}, {}, {"moments": values}

    normalizer = MomentNormalizer.fit(FakeCorpus())

    expected = np.sqrt(
        (3 * 1.0**2 + 3 * 4.0) / 6
    )

    assert np.allclose(
        normalizer.scales,
        np.full(10, expected),
    )

    assert normalizer.count == 6


def test_normalizer_roundtrip():
    state = {
        "scales": [1.0] * 10,
        "count": 100,
        "method": "train-only-rms-no-centering",
    }

    restored = MomentNormalizer.from_state_dict(state)

    assert restored.count == 100
    assert np.allclose(restored.scales, 1.0)


def test_moment_loss_zero_for_exact_prediction():
    x = torch.randn(16, 10, dtype=torch.float64)
    scales = torch.ones(10, dtype=torch.float64)

    assert moment_mse(x, x, scales).item() == 0
    assert moment_mae(x, x, scales).item() == 0


def test_moment_loss_backpropagates():
    target = torch.randn(8, 10, dtype=torch.float64)
    pred = torch.randn(
        8, 10, dtype=torch.float64, requires_grad=True
    )

    scales = torch.ones(10, dtype=torch.float64)

    loss = moment_mse(pred, target, scales)
    loss.backward()

    assert torch.isfinite(loss)
    assert pred.grad is not None
    assert torch.isfinite(pred.grad).all()
    assert pred.grad.norm() > 0


def test_metrics_report_raw_and_normalized_errors():
    target = np.ones((4, 10))
    prediction = target + 0.1
    scales = np.ones(10)

    result = evaluate_moments(
        prediction, target, scales
    )

    assert result["count"] == 4
    assert np.isclose(result["normalized_rmse"], 0.1)
    assert np.isclose(result["relative_l2_mean"], 0.1)


def test_moment_loss_rejects_nonfinite():
    target = torch.zeros(2, 10)
    prediction = torch.zeros(2, 10)
    prediction[0, 0] = float("nan")

    with pytest.raises(FloatingPointError):
        moment_mse(
            prediction,
            target,
            torch.ones(10),
        )
