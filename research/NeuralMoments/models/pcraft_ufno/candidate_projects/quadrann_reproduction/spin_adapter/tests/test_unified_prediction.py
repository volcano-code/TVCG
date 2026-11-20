from __future__ import annotations

import unittest
from types import SimpleNamespace

import torch

from research.NeuralMoments.models.pcraft_ufno.candidate_projects.quadrann_reproduction.spin_adapter.unified_prediction import (
    predict_full,
)


class _FakeModel:
    def __init__(self):
        self.shared_encoder = SimpleNamespace(
            config=SimpleNamespace(
                uq_enabled=False,
                event_validity_head_enabled=False,
            )
        )
        self.calls = []

    def evaluate_queries(self, cache, coefficients, *, create_graph, with_jacobian):
        self.calls.append(len(coefficients))
        moments = coefficients * 2.0
        jac = torch.eye(10, dtype=coefficients.dtype, device=coefficients.device)
        jac = jac.unsqueeze(0).expand(len(coefficients), -1, -1) * 2.0
        return {
            "moments": moments,
            "jacobian_coefficients": jac,
            "jacobian_q": jac,
            "geometry_id": cache.geometry_id,
            "mass_error": torch.zeros(len(coefficients), device=coefficients.device),
        }


class UnifiedPredictionContractTests(unittest.TestCase):
    def setUp(self):
        self.model = _FakeModel()
        self.cache = SimpleNamespace(
            shared_cache=SimpleNamespace(latent=torch.zeros(4)),
            geometry_id="fixture-geometry",
        )
        self.q = torch.arange(30, dtype=torch.float64).reshape(3, 10) + 1.0

    def test_chunked_shapes_and_raw_jacobian(self):
        result = predict_full(
            self.model, self.cache, self.q, query_chunk_size=2
        )
        self.assertEqual(tuple(result.moments.shape), (3, 10))
        self.assertEqual(tuple(result.jacobian_coefficients.shape), (3, 10, 10))
        self.assertTrue(torch.equal(result.moments, self.q * 2.0))
        expected = torch.eye(10, dtype=torch.float64).unsqueeze(0).expand(3, -1, -1) * 2.0
        self.assertTrue(torch.equal(result.jacobian_coefficients, expected))
        self.assertEqual(self.model.calls, [2, 1])
        self.assertEqual(result.diagnostics["chunk_count"], 2)
        self.assertTrue(result.prediction_finite.all())

    def test_missing_auxiliary_heads_are_none_not_zeros(self):
        result = predict_full(self.model, self.cache, self.q)
        self.assertIsNone(result.moment_log_std)
        self.assertIsNone(result.jacobian_log_std)
        self.assertIsNone(result.event_logits)
        self.assertIsNone(result.event_probability)
        self.assertFalse(result.uq_prediction_available)
        self.assertFalse(result.event_prediction_available)
        self.assertFalse(result.teacher_derivative_validity_available)

    def test_invalid_queries_rejected(self):
        with self.assertRaises(ValueError):
            predict_full(self.model, self.cache, torch.zeros(10))
        with self.assertRaises(ValueError):
            predict_full(self.model, self.cache, torch.zeros((0, 10)))
        with self.assertRaises(ValueError):
            predict_full(self.model, self.cache, self.q, query_chunk_size=0)

    def test_single_query_keeps_query_dimension(self):
        result = predict_full(self.model, self.cache, self.q[0])
        self.assertEqual(tuple(result.moments.shape), (1, 10))
        self.assertEqual(tuple(result.jacobian_coefficients.shape), (1, 10, 10))


if __name__ == "__main__":
    unittest.main()
