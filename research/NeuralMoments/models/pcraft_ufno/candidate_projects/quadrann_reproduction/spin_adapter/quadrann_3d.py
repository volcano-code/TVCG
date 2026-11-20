"""RLE-native, topology-aware 3D QuadrANN encoder.

Borrowed from upstream QuadrANN:
- geometric positional encoding
- mean/std message aggregation
- global mean context
- dense concatenation

Spin-specific extensions:
- one token per native RLE interval
- six-connected component restricted graph
- mass-prior positive quadrature weights
"""
from __future__ import annotations

import numpy as np
import torch
from torch import nn
from torch_geometric.nn import MessagePassing
from torch_scatter import scatter_mean

from research.NeuralMoments.models.pcraft_ufno.spin_pino.native_pcraft import (
    native_interval_components,
    build_local_neighbor_graph,
)


def positional_encoding(x, bands):
    if bands == 0:
        return x.new_empty((x.shape[0], 0))
    freq = 2.0 ** torch.arange(
        bands, device=x.device, dtype=x.dtype
    )
    phase = x.unsqueeze(-1) * np.pi * freq
    return torch.cat(
        (torch.sin(phase), torch.cos(phase)), dim=-1
    ).flatten(1)


def make_topology_graph(xyz, component_ids, neighbors=16):
    """Build directed source-to-target graph without cross-component edges."""
    xyz_np = np.asarray(xyz, dtype=np.float64)
    components = np.asarray(component_ids)

    if xyz_np.ndim != 2 or xyz_np.shape[1] != 3:
        raise ValueError("xyz must have shape [N,3]")
    if components.shape != (len(xyz_np),):
        raise ValueError("component_ids must have shape [N]")

    knn = build_local_neighbor_graph(
        xyz_np,
        neighbors=neighbors,
        component_ids=components,
    )

    n, k = knn.shape
    target = np.repeat(np.arange(n), k)
    source = knn.reshape(-1)

    edge_index = np.stack((source, target), axis=0)

    if not np.all(components[source] == components[target]):
        raise RuntimeError("graph crosses disconnected material components")

    return edge_index.astype(np.int64)


class InitialGeometryLayer(MessagePassing):
    def __init__(self, hidden_dim, pe_abs=3, pe_rel=5):
        super().__init__(aggr=["mean", "std"])
        self.pe_abs = pe_abs
        self.pe_rel = pe_rel

        in_dim = 1 + 3 + 6*pe_abs + 3 + 6*pe_rel

        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(self, xyz, density, edge_index):
        return self.propagate(
            edge_index,
            xyz=xyz,
            density=density,
            size=(len(xyz), len(xyz)),
        )

    def message(self, xyz_i, xyz_j, density_j):
        delta = xyz_j - xyz_i
        features = torch.cat((
            density_j,
            xyz_j,
            positional_encoding(xyz_j, self.pe_abs),
            delta,
            positional_encoding(delta, self.pe_rel),
        ), dim=-1)
        return self.mlp(features)

    def forward_chunk(self, xyz_source, density_source, xyz_target, density_target,
                      edge_index, *, source_count):
        """Bipartite message pass for a target chunk; source nodes stay global."""
        return self.propagate(
            edge_index,
            xyz=(xyz_source, xyz_target),
            density=(density_source, density_target),
            size=(source_count, len(xyz_target)),
        )


class FeaturePropagationLayer(MessagePassing):
    def __init__(self, hidden_dim):
        super().__init__(aggr=["mean", "std"])

        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim*6, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(self, x, edge_index, global_context):
        return self.propagate(
            edge_index,
            x=x,
            global_context=global_context,
            size=(len(x), len(x)),
        )

    def message(self, x_i, x_j, global_context_i):
        return self.mlp(torch.cat((
            x_j,
            x_j-x_i,
            global_context_i,
        ), dim=-1))

    def forward_chunk(self, x_source, x_target, global_context_source,
                      global_context_target, edge_index, *, source_count):
        """Bipartite message pass with all source states and local targets."""
        return self.propagate(
            edge_index,
            x=(x_source, x_target),
            global_context=(global_context_source, global_context_target),
            size=(source_count, len(x_target)),
        )


class SpinQuadrANN3D(nn.Module):
    """3D interval-token GNN with mass-preserving positive readout."""

    def __init__(
        self,
        hidden_dim=128,
        final_hidden_dim=256,
        num_layers=3,
        neighbors=16,
        pe_abs=3,
        pe_rel=5,
    ):
        super().__init__()

        if num_layers < 1:
            raise ValueError("num_layers must be >=1")

        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.neighbors = neighbors

        self.initial = InitialGeometryLayer(
            hidden_dim, pe_abs, pe_rel
        )

        self.layers = nn.ModuleList([
            FeaturePropagationLayer(hidden_dim)
            for _ in range(num_layers-1)
        ])

        self.output = nn.Sequential(
            nn.Linear(2*hidden_dim*num_layers, final_hidden_dim),
            nn.GELU(),
            nn.Dropout(0.5),
            nn.Linear(final_hidden_dim, final_hidden_dim//2),
            nn.GELU(),
            nn.Dropout(0.5),
            nn.Linear(final_hidden_dim//2, 1),
        )

    def forward(
        self,
        xyz,
        interval_mass,
        component_ids,
        edge_index=None,
    ):
        if xyz.ndim != 2 or xyz.shape[1] != 3:
            raise ValueError("xyz must be [N,3]")
        n = len(xyz)

        if interval_mass.shape != (n,):
            raise ValueError("interval_mass must be [N]")
        if not torch.isfinite(xyz).all():
            raise ValueError("xyz must be finite")
        if not torch.isfinite(interval_mass).all() or (interval_mass <= 0).any():
            raise ValueError("interval masses must be finite and positive")

        if edge_index is None:
            edge_np = make_topology_graph(
                xyz.detach().cpu().numpy(),
                np.asarray(component_ids),
                self.neighbors,
            )
            edge_index = torch.as_tensor(
                edge_np, dtype=torch.long, device=xyz.device
            )

        if edge_index.ndim != 2 or edge_index.shape[0] != 2:
            raise ValueError("edge_index must be [2,E]")
        if edge_index.numel() and (
            edge_index.min() < 0 or edge_index.max() >= n
        ):
            raise ValueError("edge_index out of bounds")

        center = xyz.amin(dim=0)
        span = (xyz.amax(dim=0)-center).amax().clamp_min(1e-12)
        coords = 2.0*(xyz-center)/span-1.0

        source, target = edge_index
        delta = xyz[source]-xyz[target]

        distance = torch.linalg.vector_norm(delta, dim=-1)

        mean_distance = scatter_mean(
            distance,
            target,
            dim=0,
            dim_size=n,
        )

        density = 1.0/mean_distance.clamp_min(1e-8)
        density = torch.log1p(density).unsqueeze(-1)

        features = self.initial(
            coords, density, edge_index
        )

        outputs = [features]

        for layer in self.layers:
            global_context = features.mean(
                dim=0, keepdim=True
            ).expand_as(features)

            features = layer(
                features, edge_index, global_context
            )
            outputs.append(features)

        logits = self.output(torch.cat(outputs, dim=-1)).squeeze(-1)

        # Spin-specific extension: retain actual editable total mass.
        log_prior = torch.log(interval_mass)
        probabilities = torch.softmax(logits + log_prior, dim=0)
        quadrature_mass = interval_mass.sum()*probabilities

        if not torch.isfinite(quadrature_mass).all():
            raise RuntimeError("nonfinite quadrature mass")

        return {
            "logits": logits,
            "probabilities": probabilities,
            "weights": quadrature_mass,
            "total_mass": quadrature_mass.sum(),
            "node_features": features,
        }


def interval_component_ids(columns):
    return native_interval_components(
        np.asarray(columns, dtype=np.int64)
    )
