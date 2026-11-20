from __future__ import annotations

import math
import unittest

import torch
from torch.utils.checkpoint import checkpoint

from research.NeuralMoments.models.pcraft_ufno.candidate_projects.quadrann_reproduction.spin_adapter.geometry_physics_attention import (
    GeometryPhysicsCrossAttention,
)


def reference_attention(module, coefficients, features, mass):
    dtype = features.dtype
    qnorm = coefficients / torch.linalg.vector_norm(
        coefficients, dim=-1, keepdim=True
    ).clamp_min(1e-30)
    qtoken = module.query_encoder(qnorm.to(dtype=dtype))
    q = module.query(qtoken).reshape(-1, module.heads, module.dim)
    keys = module.key(features).reshape(-1, module.heads, module.dim)
    values = module.value(features).reshape(-1, module.heads, module.dim)
    log_mass = torch.log((mass / mass.sum()).to(dtype=dtype).clamp_min(1e-30))
    scores = torch.einsum("qhd,nhd->qhn", q, keys) / math.sqrt(module.dim)
    scores = scores + log_mass.view(1, 1, -1)
    weights = torch.softmax(scores, dim=-1)
    context = torch.einsum("qhn,nhd->qhd", weights, values).reshape(-1, module.width)
    correction = module.output(torch.cat((qtoken, context), dim=-1)) * module.output_scale
    entropy = -(weights.clamp_min(1e-30) * weights.clamp_min(1e-30).log()).sum(-1).mean(-1)
    return correction, entropy


class GeometryPhysicsAttentionChunkTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(20261121)
        self.module = GeometryPhysicsCrossAttention(
            width=16, heads=4, chunk_intervals=13, hidden=24, output_scale=0.05
        ).double()
        self.module.eval()
        self.q = torch.randn(3, 10, dtype=torch.float64, requires_grad=True)
        self.features = torch.randn(47, 16, dtype=torch.float64, requires_grad=True)
        self.mass = (torch.rand(47, dtype=torch.float64) + 0.1)

    def test_streamed_attention_matches_full_reference(self):
        actual, diag = self.module(self.q, self.features, self.mass)
        expected, entropy = reference_attention(
            self.module, self.q, self.features, self.mass
        )
        self.assertTrue(torch.allclose(actual, expected, atol=2e-10, rtol=2e-10))
        self.assertTrue(torch.allclose(diag["attention_entropy"], entropy, atol=2e-10, rtol=2e-10))

    def test_streamed_attention_gradients_match_reference(self):
        actual, _ = self.module(self.q, self.features, self.mass)
        loss_actual = actual.square().sum()
        grad_actual = torch.autograd.grad(
            loss_actual, (self.q, self.features, *self.module.parameters()),
            retain_graph=False, allow_unused=True,
        )
        q_ref = self.q.detach().clone().requires_grad_(True)
        f_ref = self.features.detach().clone().requires_grad_(True)
        expected, _ = reference_attention(self.module, q_ref, f_ref, self.mass)
        loss_ref = expected.square().sum()
        grad_ref = torch.autograd.grad(
            loss_ref, (q_ref, f_ref, *self.module.parameters()),
            retain_graph=False, allow_unused=True,
        )
        for got, want in zip(grad_actual, grad_ref):
            if got is None or want is None:
                self.assertIs(got, want)
            else:
                self.assertTrue(torch.allclose(got, want, atol=2e-9, rtol=2e-9))


    def test_nonfinite_feature_is_rejected_with_chunked_validation(self):
        features = self.features.detach().clone()
        features[17, 3] = float("nan")
        diagnostic_module = GeometryPhysicsCrossAttention(
            width=16, heads=4, chunk_intervals=13, hidden=24,
            output_scale=0.05, validate_inputs=True,
        ).double()
        diagnostic_module.load_state_dict(self.module.state_dict())
        with self.assertRaisesRegex(ValueError, "chunk"):
            diagnostic_module(self.q.detach(), features, self.mass)

    def test_checkpointed_streaming_backward_is_finite(self):
        self.module.train()
        q = self.q.detach().clone().requires_grad_(True)
        features = self.features.detach().clone().requires_grad_(True)
        def attention_with_diagnostics(q_input, feature_input, mass_input):
            correction, diagnostics = self.module(q_input, feature_input, mass_input)
            return correction, diagnostics["attention_entropy"]
        correction, _ = checkpoint(
            attention_with_diagnostics, q, features, self.mass,
            use_reentrant=False,
        )
        # Include a query derivative so the test exercises the outer
        # checkpointed graph used by UnifiedM80SQNO's Jacobian supervision.
        jac = torch.autograd.grad(
            correction.sum(), q, create_graph=True, retain_graph=True
        )[0]
        loss = correction.square().mean() + jac.square().mean()
        loss.backward()
        self.assertTrue(torch.isfinite(loss))
        self.assertIsNotNone(features.grad)
        self.assertTrue(torch.isfinite(features.grad).all())
        for parameter in self.module.parameters():
            if parameter.grad is not None:
                self.assertTrue(torch.isfinite(parameter.grad).all())


if __name__ == "__main__":
    unittest.main()
