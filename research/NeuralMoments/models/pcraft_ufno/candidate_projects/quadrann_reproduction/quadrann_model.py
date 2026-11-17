"""Isolated QuadrANN model implementation.

Adapted from upstream:
kesmarag/QuadrANN
commit: 111dd611aaeac1fb98c76385706b003ea8ef94be

The upstream training script is intentionally not imported.
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_scatter import scatter_mean
from torch_geometric.nn import (
    knn_graph,
    MessagePassing,
    global_mean_pool,
)


def positional_encoding(pos, num_bands, base=2.0, scale_type="linear"):
    if num_bands == 0:
        return torch.empty(
            pos.shape[0], 0, device=pos.device, dtype=pos.dtype
        )

    if scale_type == "linear":
        freq = torch.linspace(
            1.0, base ** (num_bands - 1),
            steps=num_bands, device=pos.device, dtype=pos.dtype
        )
    elif scale_type == "power":
        freq = base ** torch.arange(
            num_bands, device=pos.device, dtype=pos.dtype
        )
    else:
        raise ValueError(scale_type)

    projections = pos.unsqueeze(-1) * np.pi * freq
    return torch.cat(
        [torch.sin(projections), torch.cos(projections)], dim=-1
    ).flatten(start_dim=1)


def calculate_density(coords, edge_index):
    row, col = edge_index
    neighbor_coords = coords[col]

    centroids = scatter_mean(
        neighbor_coords, row, dim=0, dim_size=coords.size(0)
    )
    edge_centroids = centroids[row]
    distances = torch.norm(
        neighbor_coords - edge_centroids, p=2, dim=-1
    )
    mean_distance = scatter_mean(
        distances, row, dim=0, dim_size=coords.size(0)
    )
    return 1.0 / (mean_distance + 1e-8).unsqueeze(-1)


class InitialGeometricLayer(MessagePassing):
    def __init__(self, mlp, pe_abs, pe_rel, aggr):
        super().__init__(aggr=aggr)
        self.mlp = mlp
        self.pe_abs = pe_abs
        self.pe_rel = pe_rel

    def forward(self, coords, density, edge_index):
        return self.propagate(
            edge_index, coords=coords, density=density
        )

    def message(self, coords_i, coords_j, density_j):
        def pe(tensor, bands):
            return positional_encoding(
                tensor, bands, base=2.0, scale_type="power"
            )

        coord_diff = coords_j - coords_i

        features = [
            density_j,
            coords_j,
            pe(coords_j, self.pe_abs),
            coord_diff,
            pe(coord_diff, self.pe_rel),
        ]
        return self.mlp(torch.cat(features, dim=-1))


class FeaturePropagationLayer(MessagePassing):
    def __init__(self, mlp, aggr):
        super().__init__(aggr=aggr)
        self.mlp = mlp

    def forward(self, x, edge_index, global_context):
        return self.propagate(
            edge_index, x=x, global_context=global_context
        )

    def message(self, x_i, x_j, global_context_i):
        return self.mlp(
            torch.cat([x_j, x_j - x_i, global_context_i], dim=-1)
        )


class QuadrANN(nn.Module):
    def __init__(
        self,
        dim,
        num_gnn_layers,
        k_neighbors,
        pe_abs,
        pe_rel,
        hidden_dim,
        final_hidden_dim,
    ):
        super().__init__()

        self.k_neighbors = k_neighbors
        self.dim = dim

        aggregators = ["mean", "std"]
        num_aggregators = len(aggregators)

        mlp1_in_dim = (
            1
            + dim + dim * 2 * pe_abs
            + dim + dim * 2 * pe_rel
        )

        mlp1 = nn.Sequential(
            nn.Linear(mlp1_in_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        self.initial_layer = InitialGeometricLayer(
            mlp1, pe_abs, pe_rel, aggr=aggregators
        )

        self.feature_layers = nn.ModuleList()

        for _ in range(num_gnn_layers - 1):
            mlp_in_dim = hidden_dim * num_aggregators * 3

            mlp = nn.Sequential(
                nn.Linear(mlp_in_dim, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, hidden_dim),
            )

            self.feature_layers.append(
                FeaturePropagationLayer(mlp, aggr=aggregators)
            )

        final_in_dim = (
            hidden_dim * num_aggregators * num_gnn_layers
        )

        self.final_mlp = nn.Sequential(
            nn.Linear(final_in_dim, final_hidden_dim),
            nn.GELU(),
            nn.Dropout(0.5),
            nn.Linear(final_hidden_dim, final_hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(0.5),
            nn.Linear(final_hidden_dim // 2, 1),
        )

    def forward(self, x):
        batch_size, num_points, coord_dim = x.shape

        batch = torch.arange(
            batch_size, device=x.device
        ).repeat_interleave(num_points)

        coords = 2.0 * (x.reshape(-1, coord_dim) - 0.5)

        edge_index = knn_graph(
            coords, k=self.k_neighbors, batch=batch
        )

        density = calculate_density(coords, edge_index)

        outputs = []

        features = self.initial_layer(
            coords, density, edge_index
        )
        outputs.append(features)

        for layer in self.feature_layers:
            global_context = global_mean_pool(
                features, batch
            )[batch]

            features = layer(
                features, edge_index, global_context
            )
            outputs.append(features)

        logits = self.final_mlp(
            torch.cat(outputs, dim=-1)
        )

        return F.softmax(
            logits.reshape(batch_size, num_points), dim=-1
        )
