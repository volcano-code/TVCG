from __future__ import annotations

import unittest

import torch

from research.NeuralMoments.models.pcraft_ufno.spin_pino.native_pcraft import (
    PhysicsStateSlicing,
)


class PhysicsStateSlicingChunkTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(20261120)
        self.module = PhysicsStateSlicing(width=16, n_slices=8, n_heads=4)
        self.features = torch.randn(73, 16, requires_grad=True)
        self.mass = torch.rand(73) + 0.1

    def test_chunked_and_single_chunk_forward_match(self):
        self.module.eval()
        with torch.no_grad():
            whole, whole_metrics = self.module(
                self.features, self.mass, point_chunk_size=1000
            )
            chunked, chunked_metrics = self.module(
                self.features, self.mass, point_chunk_size=11
            )
        self.assertTrue(torch.allclose(whole, chunked, atol=2e-6, rtol=2e-6))
        self.assertAlmostEqual(
            whole_metrics["transolver_slice_mass_sum_error"],
            chunked_metrics["transolver_slice_mass_sum_error"],
            places=6,
        )

    def test_chunked_training_backward_is_finite(self):
        self.module.train()
        output, metrics = self.module(
            self.features, self.mass, point_chunk_size=13
        )
        loss = output.square().mean()
        loss.backward()
        self.assertTrue(torch.isfinite(loss))
        self.assertIsNotNone(self.features.grad)
        self.assertTrue(torch.isfinite(self.features.grad).all())
        for parameter in self.module.parameters():
            if parameter.grad is not None:
                self.assertTrue(torch.isfinite(parameter.grad).all())
        self.assertEqual(metrics["transolver_point_chunk_size"], 13.0)

    def test_rejects_nonpositive_chunk_size(self):
        with self.assertRaises(ValueError):
            self.module(self.features, self.mass, point_chunk_size=0)


if __name__ == "__main__":
    unittest.main()
