"""SQNO v1.1: P-CRAFT + topology-aware QuadrANN fusion."""
from __future__ import annotations

import numpy as np
import torch
from torch import nn
from torch_scatter import scatter_mean

from .pcraft_feature_adapter import PcraftFeatureAdapter
from .quadrann_3d import (
    SpinQuadrANN3D,
    make_topology_graph,
    interval_component_ids,
)
from .rle_quadrature import rle_gauss_quadrature
from .spin_moment_operator import SpinMomentOperator


class SpinQuadrANNV11(nn.Module):
    def __init__(
        self,
        *,
        preset="large_ms",
        hidden_dim=32,
        final_hidden_dim=128,
        num_layers=3,
        neighbors=16,
        temperature=0.04,
        quadrature_order=2,
        chunk_intervals=512,
        chunk_nodes=65536,
        pcraft_width=None,
        gnn_chunk_size=8192,
    ):
        super().__init__()
        self.external_pcraft_width = None if pcraft_width is None else int(pcraft_width)
        if self.external_pcraft_width is not None:
            if self.external_pcraft_width < 1:
                raise ValueError("pcraft_width must be positive")
            self.pcraft = None
            self.pcraft_width = self.external_pcraft_width
        else:
            self.pcraft = PcraftFeatureAdapter(
                preset=preset, min_native_resolution=512,
                chunk_intervals=chunk_intervals,
            )
            self.pcraft_width = self.pcraft.operator.config.interval_width

        self.hidden_dim = int(hidden_dim)
        self.final_hidden_dim = int(final_hidden_dim)
        self.num_layers = int(num_layers)
        self.neighbors = int(neighbors)
        self.chunk_intervals = int(chunk_intervals)
        self.quadrature_order = int(quadrature_order)
        self.chunk_nodes = int(chunk_nodes)
        self.gnn_chunk_size = int(gnn_chunk_size)
        if self.gnn_chunk_size < 1:
            raise ValueError("gnn_chunk_size must be positive")

        self.gnn = SpinQuadrANN3D(
            hidden_dim=hidden_dim,
            final_hidden_dim=final_hidden_dim,
            num_layers=num_layers,
            neighbors=neighbors,
        )

        # v1.1 uses GNN layer features directly.
        # Remove the unused original QuadrANN Softmax head.
        self.gnn.output = nn.Identity()

        fusion_dim = (
            2 * hidden_dim * num_layers
            + 4 * hidden_dim
            + 2 * self.pcraft_width
        )

        self.fusion = nn.Sequential(
            nn.LayerNorm(fusion_dim),
            nn.Linear(fusion_dim, final_hidden_dim),
            nn.GELU(),
            nn.Linear(final_hidden_dim, final_hidden_dim),
            nn.GELU(),
        )

        self.weight_head = nn.Sequential(
            nn.Linear(final_hidden_dim, final_hidden_dim // 2),
            nn.GELU(),
            nn.Linear(final_hidden_dim // 2, 1),
        )

        self.moment_operator = SpinMomentOperator(
            temperature=temperature,
            chunk_nodes=chunk_nodes,
        )

        self.quadrature_order = quadrature_order

    def _gnn_features(self, xyz, mass, components):
        """Chunk QuadrANN message passing over target intervals.

        The topology graph retains global source indices, while each target
        chunk is remapped to local indices. Global mean context is computed
        across all nodes between message-passing layers, preserving the
        original full-graph semantics without materializing E-sized message
        activations for all 1M+ intervals at once.
        """
        n = len(xyz)
        chunk_size = self.gnn_chunk_size
        edge_np = make_topology_graph(
            xyz.detach().cpu().numpy(),
            np.asarray(components),
            neighbors=self.gnn.neighbors,
        )
        edges_per_target = edge_np.shape[1] // n
        if edges_per_target * n != edge_np.shape[1]:
            raise RuntimeError("topology graph must have a fixed neighbor count per target")

        center = xyz.amin(dim=0)
        span = (xyz.amax(dim=0) - center).amax().clamp_min(1e-12)
        coords = 2 * (xyz - center) / span - 1

        initial_chunks = []
        for start in range(0, n, chunk_size):
            stop = min(start + chunk_size, n)
            edge_slice = edge_np[:, start * edges_per_target:stop * edges_per_target].copy()
            edge_slice[1] -= start
            edge = torch.as_tensor(edge_slice, dtype=torch.long, device=xyz.device)
            source, target_local = edge
            xyz_target = xyz[start:stop]
            distance = torch.linalg.vector_norm(
                xyz[source] - xyz_target[target_local], dim=-1
            )
            mean_distance = scatter_mean(
                distance, target_local, dim=0, dim_size=stop-start
            )
            density_target = torch.log1p(
                1.0 / mean_distance.clamp_min(1e-8)
            ).unsqueeze(-1)
            initial_chunks.append((start, stop, density_target))

        # Assemble per-node density in chunks, then perform initial messages.
        density = torch.cat([item[2] for item in initial_chunks], dim=0)
        first_features = []
        for start, stop, _ in initial_chunks:
            edge_slice = edge_np[:, start * edges_per_target:stop * edges_per_target].copy()
            edge_slice[1] -= start
            edge = torch.as_tensor(edge_slice, dtype=torch.long, device=xyz.device)
            features = self.gnn.initial.forward_chunk(
                coords, density, coords[start:stop], density[start:stop],
                edge, source_count=n,
            )
            first_features.append(features)
        features = torch.cat(first_features, dim=0)
        layer_outputs = [features]

        for layer in self.gnn.layers:
            global_mean = features.mean(dim=0, keepdim=True)
            global_context = global_mean.expand(n, -1)
            updated_chunks = []
            for start, stop, _ in initial_chunks:
                edge_slice = edge_np[:, start * edges_per_target:stop * edges_per_target].copy()
                edge_slice[1] -= start
                edge = torch.as_tensor(edge_slice, dtype=torch.long, device=xyz.device)
                updated = layer.forward_chunk(
                    features, features[start:stop], global_context,
                    global_context[start:stop], edge, source_count=n,
                )
                updated_chunks.append(updated)
            features = torch.cat(updated_chunks, dim=0)
            layer_outputs.append(features)

        return torch.cat(layer_outputs, dim=-1), features

    def forward(
        self,
        *,
        geometry,
        coefficients,
        component_ids=None,
        with_jacobian=True,
    ):
        if coefficients.shape != (10,):
            raise ValueError("coefficients must be [10]")

        device = coefficients.device
        dtype = coefficients.dtype

        if "xyz" not in geometry:
            from research.NeuralMoments.models.pcraft_ufno.spin_pino.native_pcraft import (
                geometry_from_columns,
            )
            geometry = geometry_from_columns(geometry)

        # Branch A: multi-scale P-CRAFT.
        if self.pcraft is None:
            raise RuntimeError("this SQNO instance requires externally supplied P-CRAFT features")
        pcraft = self.pcraft(geometry)

        # Branch B: native RLE topology-aware QuadrANN.
        xyz = torch.as_tensor(
            np.asarray(geometry["xyz"]),
            device=device,
            dtype=dtype,
        )

        interval_mass = pcraft["interval_mass"].to(
            device=device, dtype=dtype
        )

        if component_ids is None:
            component_ids = interval_component_ids(
                geometry["columns"]
            )

        component_ids = np.asarray(component_ids)

        if component_ids.shape != (len(xyz),):
            raise ValueError("component_ids shape mismatch")

        q_features, final_features = self._gnn_features(
            xyz, interval_mass, component_ids
        )

        # Component-level pooled context.
        _, inverse = np.unique(
            component_ids, return_inverse=True
        )

        inverse = torch.as_tensor(
            inverse, device=device, dtype=torch.long
        )

        n_components = int(inverse.max().item()) + 1

        component_context = torch.zeros(
            n_components,
            final_features.shape[-1],
            device=device,
            dtype=dtype,
        )

        component_context.index_add_(
            0, inverse, final_features
        )

        component_counts = torch.bincount(
            inverse, minlength=n_components
        ).to(dtype).clamp_min(1)

        component_context = (
            component_context / component_counts[:, None]
        )

        local_component_context = component_context[inverse]

        p_local = pcraft["local_features"].to(dtype)
        p_global = pcraft["global_context"].to(dtype)
        p_global = p_global.unsqueeze(0).expand(len(xyz), -1)

        fused = self.fusion(torch.cat((
            q_features,
            final_features,
            p_local,
            p_global,
            local_component_context,
        ), dim=-1))

        logits = self.weight_head(fused).squeeze(-1)

        # Exactly one mass-prior Softmax.
        probabilities = torch.softmax(
            logits.float() + interval_mass.log().float(),
            dim=0,
        ).to(dtype)

        learned_mass = interval_mass.sum() * probabilities

        # Native RLE Gauss nodes.
        quadrature = rle_gauss_quadrature(
            geometry, order=self.quadrature_order
        )

        node_xyz = torch.as_tensor(
            quadrature["xyz"], device=device, dtype=dtype
        )

        base_weights = torch.as_tensor(
            quadrature["weights"], device=device, dtype=dtype
        )

        ids = torch.as_tensor(
            quadrature["interval_ids"],
            device=device,
            dtype=torch.long,
        )

        base_interval_mass = torch.as_tensor(
            np.bincount(
                np.asarray(quadrature["interval_ids"]),
                weights=np.asarray(quadrature["weights"]),
                minlength=len(interval_mass),
            ),
            device=device,
            dtype=dtype,
        )

        local_fraction = (
            base_weights / base_interval_mass[ids]
        )

        node_weights = learned_mass[ids] * local_fraction

        mass_error = (
            node_weights.sum() - interval_mass.sum()
        ).abs()

        if not torch.isfinite(node_weights).all():
            raise FloatingPointError("nonfinite node weights")

        if (node_weights <= 0).any():
            raise FloatingPointError("nonpositive node weights")

        moments = self.moment_operator.evaluate(
            node_xyz,
            node_weights,
            coefficients,
            with_jacobian=with_jacobian,
        )

        return {
            **moments,
            "logits": logits,
            "probabilities": probabilities,
            "learned_interval_mass": learned_mass,
            "node_weights": node_weights,
            "interval_ids": ids,
            "pcraft_local_features": p_local,
            "pcraft_global_context": p_global[0],
            "quadrann_features": q_features,
            "component_context": local_component_context,
            "total_mass_error": mass_error,
            "n_intervals": len(interval_mass),
            "n_nodes": len(node_xyz),
            "global_scale_weights": pcraft["global_scale_weights"],
        }
