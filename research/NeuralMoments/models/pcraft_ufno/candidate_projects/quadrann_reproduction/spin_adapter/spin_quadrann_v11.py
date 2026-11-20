"""SQNO v1.1: P-CRAFT + topology-aware QuadrANN fusion."""
from __future__ import annotations

import numpy as np
import torch
from torch import nn
from torch_scatter import scatter_mean

from .pcraft_feature_adapter import PcraftFeatureAdapter
from .memory_profile import profile_count
from research.NeuralMoments.models.pcraft_ufno.spin_pino.native_pcraft import (
    build_local_neighbor_graph, checkpoint_stage,
)
from research.NeuralMoments.models.pcraft_ufno.spin_pino.geometry_tiles import neighbor_tiles
from .quadrann_3d import (
    SpinQuadrANN3D,
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
        gnn_chunk_size=1024,
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

    def _gnn_features(self, xyz, mass, components, *, return_layer_features=False, graph=None):
        """Chunk QuadrANN message passing over target intervals.

        The topology graph retains global source indices, while each target
        chunk is remapped to local indices. Global mean context is computed
        across all nodes between message-passing layers, preserving the
        original full-graph semantics without materializing E-sized message
        activations for all 1M+ intervals at once.
        """
        n = len(xyz)
        chunk_size = self.gnn_chunk_size
        if graph is None:
            graph = build_local_neighbor_graph(xyz.detach().cpu().numpy(), self.gnn.neighbors, components)
        tiles = list(neighbor_tiles(graph, chunk_size, xyz.device))
        xyz_parts = xyz.split(chunk_size)
        center = xyz.amin(dim=0)
        span = (xyz.amax(dim=0) - center).amax().clamp_min(1e-12)
        coords = ((xyz - center) * 2 / span - 1).split(chunk_size)
        edges, densities = [], []
        for i, tile in enumerate(tiles):
            indices = tile.indices.long()
            source = indices.reshape(-1)
            target = torch.arange(len(indices), device=xyz.device).repeat_interleave(indices.shape[1])
            edges.append(torch.stack((source, target)))
            xyz_source = tile.gather_source_table(xyz_parts)
            distance = torch.linalg.vector_norm(xyz_source[source] - xyz_parts[i][target], dim=-1)
            mean_distance = scatter_mean(distance, target, dim=0, dim_size=len(indices))
            densities.append(torch.log1p(1 / mean_distance.clamp_min(1e-8)).unsqueeze(-1))

        first = []
        for i, tile in enumerate(tiles):
            xyz_source = tile.gather_source_table(coords)
            density_source = tile.gather_source_table(densities)
            def initial_chunk(xs, ds, xt, dt, edge, layer=self.gnn.initial):
                profile_count(gnn_initial_chunks=1)
                return layer.forward_chunk(xs, ds, xt, dt, edge, source_count=len(xs))
            args = (xyz_source, density_source, coords[i], densities[i], edges[i])
            first.append(checkpoint_stage("sqno_gnn_initial_recompute",initial_chunk,*args)
                         if self.training and torch.is_grad_enabled() else initial_chunk(*args))
        features = torch.cat(first)
        del first
        layer_outputs = [features]
        for layer in self.gnn.layers:
            global_mean = features.mean(dim=0, keepdim=True)
            parts = features.split(chunk_size)
            updated = []
            for i, tile in enumerate(tiles):
                def propagate(target, mean, edge, *sources, layer=layer):
                    profile_count(gnn_propagation_chunks=1)
                    src = sources[0] if len(sources)==1 else torch.cat(sources)
                    return layer.forward_chunk(src,target,mean.expand_as(src),mean.expand_as(target),
                                               edge,source_count=len(src))
                args = (parts[i],global_mean,edges[i],*(parts[j] for j in tile.sources))
                updated.append(checkpoint_stage("sqno_gnn_layer_recompute",propagate,*args)
                               if self.training and torch.is_grad_enabled() else propagate(*args))
            features = torch.cat(updated)
            layer_outputs.append(features)

        if return_layer_features:
            return layer_outputs, features
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
