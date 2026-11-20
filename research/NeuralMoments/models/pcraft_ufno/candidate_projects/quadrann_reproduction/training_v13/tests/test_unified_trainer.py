from __future__ import annotations

import unittest
from types import SimpleNamespace

import numpy as np

from research.NeuralMoments.models.pcraft_ufno.candidate_projects.quadrann_reproduction.training_v13.train_unified import (
    MomentNormalizer,
)


class _Corpus:
    def __init__(self):
        self.splits = {
            "train": [
                {"id": "train-a"}, {"id": "train-b"},
            ],
            "val": [{"id": "val-only"}],
            "test": [{"id": "test-only"}],
        }
        self.labels = {
            "train-a": np.asarray([[3.0] * 10, [4.0] * 10]),
            "train-b": np.asarray([[0.0] * 10]),
            "val-only": np.asarray([[1000.0] * 10]),
            "test-only": np.asarray([[2000.0] * 10]),
        }

    def load(self, row):
        return {}, {
            "valid_mask": np.ones(len(self.labels[row["id"]]), dtype=bool),
            "moments": self.labels[row["id"]],
        }


class UnifiedTrainerNormalizerTests(unittest.TestCase):
    def test_fit_uses_train_only_rms_without_centering(self):
        normalizer = MomentNormalizer.fit(_Corpus())
        # sqrt((3^2 + 4^2 + 0^2) / 3) = sqrt(25/3)
        self.assertTrue(np.allclose(normalizer.scales, np.full(10, np.sqrt(25.0 / 3.0))))
        self.assertEqual(normalizer.count, 3)

    def test_state_dict_is_explicit(self):
        normalizer = MomentNormalizer(np.ones(10), 17)
        state = normalizer.state_dict()
        self.assertEqual(state["method"], "train-only-rms-no-centering")
        self.assertEqual(state["count"], 17)
        self.assertEqual(len(state["scales"]), 10)

    def test_rejects_invalid_scales(self):
        with self.assertRaises(ValueError):
            MomentNormalizer(np.zeros(10), 1)
        with self.assertRaises(ValueError):
            MomentNormalizer(np.ones(9), 1)


if __name__ == "__main__":
    unittest.main()
