"""Integrated RLE-native Spin QuadrANN operator.

The model composes:
1. Native RLE Gaussian quadrature.
2. Topology-aware 3D QuadrANN.
3. Differentiable mass-preserving interval redistribution.
4. Ten-dimensional smooth convex moment readout.

Geometry-dependent graph construction is intentionally detached.
Gradients flow from the moment loss through learned interval masses
back into all trainable GNN parameters.
"""
from __future__ import annotations

import numpy as np
import torch
from torch import nn

from .rle_quadrature import rle_gauss_quadrature
from .quadrann_3d import SpinQuadrANN3D, interval_component_ids
from .spin_moment_operator import SpinMomentOperator


class SpinQuadrANN(nn.Module):
    def __init__(
        self,
        *,
        hidden_dim=128,
        final_hidden_dim=256,
        num_layers=3,
        neighbors=16,
        temperature=0.04,
        quadrature_order=2,
        chunk_nodes=65536,
    ):
        super().__init__()

        if quadrature_order < 2:
            raise ValueError("quadrature_order must be >= 2")

        self.encoder = SpinQuadrANN3D(
            hidden_dim=hidden_dim,
            final_hidden_dim=final_hidden_dim,
            num_layers=num_layers,
            neighbors=neighbors,
        )

        self.moment_operator = SpinMomentOperator(
            temperature=temperature,
            chunk_nodes=chunk_nodes,
        )

        self.quadrature_order = quadrature_order
        self.chunk_nodes = chunk_nodes

    def forward(
        self,
        *,
        geometry: dict,
        coefficients: torch.Tensor,
        component_ids=None,
        edge_index=None,
        with_jacobian=True,
    ):
        if coefficients.shape != (10,):
            raise ValueError("coefficients must have shape [10]")

        device = coefficients.device
        dtype = coefficients.dtype

        # Static geometric quadrature. Geometry is not a trainable input.
        quadrature = rle_gauss_quadrature(
            geometry,
            order=self.quadrature_order,
        )

        xyz = torch.as_tensor(
            quadrature["xyz"],
            device=device,
            dtype=dtype,
        )

        base_weights = torch.as_tensor(
            quadrature["weights"],
            device=device,
            dtype=dtype,
        )

        interval_ids_np = np.asarray(
            quadrature["interval_ids"],
            dtype=np.int64,
        )

        interval_mass = torch.as_tensor(
            quadrature["interval_mass"],
            device=device,
            dtype=dtype,
        )

        if component_ids is None:
            component_ids = interval_component_ids(
                geometry["columns"]
            )

        component_ids = np.asarray(component_ids)

        if component_ids.shape != (len(interval_mass),):
            raise ValueError("component_ids must match RLE intervals")

        if len(xyz) != len(base_weights):
            raise RuntimeError("invalid quadrature arrays")

        # Geometry encoder predicts positive interval masses.
        encoded = self.encoder(
            xyz=torch.as_tensor(
                np.asarray(quadrature["interval_centers"]),
                device=device,
                dtype=dtype,
            ),
            interval_mass=interval_mass,
            component_ids=component_ids,
            edge_index=edge_index,
        )

        learned_mass = encoded["weights"]

        # Differentiable redistribution:
        # preserve the local Gauss rule while changing interval mass.
        ids = torch.as_tensor(
            interval_ids_np,
            device=device,
            dtype=torch.long,
        )

        base_interval_mass = torch.as_tensor(
            np.bincount(
                interval_ids_np,
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

        if not torch.isfinite(node_weights).all():
            raise FloatingPointError("nonfinite node weights")

        if (node_weights <= 0).any():
            raise FloatingPointError("nonpositive node weights")

        if not torch.allclose(
            node_weights.sum(),
            interval_mass.sum(),
            rtol=1e-6,
            atol=1e-8,
        ):
            raise RuntimeError("total mass conservation failed")

        # The Moment operator receives Torch tensors directly.
        # No NumPy conversion occurs on the learned-weight path.
        moments = self.moment_operator.evaluate(
            xyz,
            node_weights,
            coefficients,
            with_jacobian=with_jacobian,
        )

        return {
            **moments,
            "logits": encoded["logits"],
            "probabilities": encoded["probabilities"],
            "learned_interval_mass": learned_mass,
            "interval_mass": interval_mass,
            "node_weights": node_weights,
            "node_features": encoded["node_features"],
            "xyz": xyz,
            "interval_ids": ids,
            "total_mass_error": (
                node_weights.sum() - interval_mass.sum()
            ).abs(),
            "quadrature_order": self.quadrature_order,
            "n_intervals": len(interval_mass),
            "n_nodes": len(xyz),
        }
