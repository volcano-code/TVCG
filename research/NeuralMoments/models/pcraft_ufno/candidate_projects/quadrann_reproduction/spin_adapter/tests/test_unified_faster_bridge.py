from __future__ import annotations

import unittest
from types import SimpleNamespace

import torch

from research.NeuralMoments.models.pcraft_ufno.candidate_projects.quadrann_reproduction.spin_adapter.unified_faster_bridge import (
    make_unified_moment_callback,
)


class _AnalyticUnifiedModel:
    def evaluate_queries(self, cache, coefficients, *, create_graph, with_jacobian):
        self.last_create_graph = create_graph
        q = coefficients
        moments = q.square() + 0.25 * q
        jac = torch.diag_embed(2.0 * q + 0.25)
        return {
            "moments": moments,
            "jacobian_coefficients": jac,
        }


class UnifiedFasterBridgeTests(unittest.TestCase):
    def test_callback_preserves_graph_for_kkt_second_derivatives(self):
        model = _AnalyticUnifiedModel()
        cache = SimpleNamespace(shared_cache=SimpleNamespace(latent=torch.zeros(1)))
        callback = make_unified_moment_callback(model, cache, create_graph=True)
        beta = torch.linspace(0.1, 1.0, 10, dtype=torch.float64).requires_grad_(True)
        moments = callback(beta)
        self.assertEqual(tuple(moments.shape), (10,))
        self.assertTrue(model.last_create_graph)
        first = torch.autograd.grad(moments.sum(), beta, create_graph=True)[0]
        second = torch.autograd.grad(first.sum(), beta)[0]
        self.assertTrue(torch.allclose(first, 2.0 * beta + 0.25))
        self.assertTrue(torch.allclose(second, torch.full_like(beta, 2.0)))

    def test_callback_rejects_wrong_beta_shape(self):
        model = _AnalyticUnifiedModel()
        cache = SimpleNamespace(shared_cache=SimpleNamespace(latent=torch.zeros(1)))
        callback = make_unified_moment_callback(model, cache, create_graph=True)
        with self.assertRaises(ValueError):
            callback(torch.ones(9, dtype=torch.float64))

    def test_callback_requires_higher_order_graph(self):
        model = _AnalyticUnifiedModel()
        cache = SimpleNamespace(shared_cache=SimpleNamespace(latent=torch.zeros(1)))
        with self.assertRaises(ValueError):
            make_unified_moment_callback(model, cache, create_graph=False)


if __name__ == "__main__":
    unittest.main()
