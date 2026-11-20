from __future__ import annotations

import unittest

import numpy as np
import torch
from torch_scatter import scatter_mean

from research.NeuralMoments.models.pcraft_ufno.candidate_projects.quadrann_reproduction.spin_adapter.spin_quadrann_v11 import (
    SpinQuadrANNV11,
)
from research.NeuralMoments.models.pcraft_ufno.candidate_projects.quadrann_reproduction.spin_adapter.quadrann_3d import (
    make_topology_graph,
)


class QuadrANNChunkedGNNTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(20261120)
        self.model = SpinQuadrANNV11(
            preset="smoke",
            hidden_dim=8,
            final_hidden_dim=16,
            num_layers=3,
            neighbors=3,
            pcraft_width=8,
            gnn_chunk_size=7,
        )
        self.xyz = torch.randn(29, 3, dtype=torch.float32, requires_grad=True)
        self.mass = torch.rand(29) + 0.1
        self.components = np.zeros(29, dtype=np.int64)

    def _full_reference(self):
        xyz = self.xyz
        n = len(xyz)
        edge_np = make_topology_graph(
            xyz.detach().cpu().numpy(), self.components,
            neighbors=self.model.gnn.neighbors,
        )
        edge = torch.as_tensor(edge_np, dtype=torch.long, device=xyz.device)
        center = xyz.amin(dim=0)
        span = (xyz.amax(dim=0) - center).amax().clamp_min(1e-12)
        coords = 2 * (xyz - center) / span - 1
        source, target = edge
        distance = torch.linalg.vector_norm(xyz[source] - xyz[target], dim=-1)
        mean_distance = scatter_mean(distance, target, dim=0, dim_size=n)
        density = torch.log1p(1.0 / mean_distance.clamp_min(1e-8)).unsqueeze(-1)
        features = self.model.gnn.initial(coords, density, edge)
        outputs = [features]
        for layer in self.model.gnn.layers:
            global_context = features.mean(dim=0, keepdim=True).expand_as(features)
            features = layer(features, edge, global_context)
            outputs.append(features)
        return torch.cat(outputs, dim=-1), features

    def test_chunked_matches_full_graph_forward(self):
        self.model.eval()
        expected_q, expected_final = self._full_reference()
        actual_q, actual_final = self.model._gnn_features(
            self.xyz, self.mass, self.components
        )
        self.assertTrue(torch.allclose(actual_q, expected_q, atol=2e-6, rtol=2e-6))
        self.assertTrue(torch.allclose(actual_final, expected_final, atol=2e-6, rtol=2e-6))

    def test_chunked_gnn_backward_is_finite(self):
        q_features, final_features = self.model._gnn_features(
            self.xyz, self.mass, self.components
        )
        loss = q_features.square().mean() + final_features.square().mean()
        loss.backward()
        self.assertTrue(torch.isfinite(loss))
        self.assertIsNotNone(self.xyz.grad)
        self.assertTrue(torch.isfinite(self.xyz.grad).all())
        for parameter in self.model.gnn.parameters():
            if parameter.grad is not None:
                self.assertTrue(torch.isfinite(parameter.grad).all())


    def test_layer_feature_mode_matches_concatenated_contract(self):
        self.model.eval()
        joined, final_a = self.model._gnn_features(
            self.xyz, self.mass, self.components
        )
        layers, final_b = self.model._gnn_features(
            self.xyz, self.mass, self.components,
            return_layer_features=True,
        )
        self.assertEqual(len(layers), self.model.num_layers + 1)
        self.assertTrue(torch.allclose(joined, torch.cat(layers, dim=-1), atol=2e-6, rtol=2e-6))
        self.assertTrue(torch.allclose(final_a, final_b, atol=2e-6, rtol=2e-6))

    def test_rejects_invalid_chunk_size(self):
        with self.assertRaises(ValueError):
            SpinQuadrANNV11(
                preset="smoke", hidden_dim=8, final_hidden_dim=16,
                num_layers=2, neighbors=2, pcraft_width=8,
                gnn_chunk_size=0,
            )


if __name__ == "__main__":
    unittest.main()
