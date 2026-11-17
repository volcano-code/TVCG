"""SQNO v1.2: geometry-query decoupled model.

Geometry is encoded once, then reused for batched coefficient queries.
The geometry cache deliberately retains its autograd graph during training.
"""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np
import torch
from torch import nn

from .spin_quadrann_v11 import SpinQuadrANNV11
from .batched_moment_operator import BatchedSpinMomentOperator
from .quadrann_3d import interval_component_ids
from .rle_quadrature import rle_gauss_quadrature


@dataclass
class SQNOGeometryCache:
    geometry_id: str
    node_xyz: torch.Tensor
    node_weights: torch.Tensor
    interval_mass: torch.Tensor
    fixed_moments: torch.Tensor
    interval_ids: torch.Tensor
    logits: torch.Tensor
    probabilities: torch.Tensor
    pcraft_local_features: torch.Tensor
    pcraft_global_context: torch.Tensor
    quadrann_features: torch.Tensor
    component_context: torch.Tensor
    global_scale_weights: torch.Tensor
    n_intervals: int
    n_nodes: int


class SpinQuadrANNV12(SpinQuadrANNV11):
    """Geometry encoder plus batched query evaluator."""

    def __init__(self, *args, query_chunk=16, **kwargs):
        super().__init__(*args, **kwargs)

        self.batched_moment_operator = BatchedSpinMomentOperator(
            temperature=self.moment_operator.temperature,
            chunk_nodes=self.moment_operator.chunk_nodes,
            chunk_queries=query_chunk,
        )

        self.geometry_encode_calls = 0
        self.query_evaluation_calls = 0

    def encode_geometry(self, geometry, component_ids=None):
        """Encode one geometry and retain a differentiable cache."""

        self.geometry_encode_calls += 1

        if "xyz" not in geometry:
            from research.NeuralMoments.models.pcraft_ufno.spin_pino.native_pcraft import (
                geometry_from_columns,
            )
            geometry = geometry_from_columns(geometry)

        # P-CRAFT branch: exactly one multiscale encoding.
        pcraft = self.pcraft(geometry)

        parameter = next(self.parameters())
        device = parameter.device
        dtype = parameter.dtype

        xyz = torch.as_tensor(
            np.asarray(geometry["xyz"]).copy(),
            device=device,
            dtype=dtype,
        )

        interval_mass = pcraft["interval_mass"].to(
            device=device,
            dtype=dtype,
        )

        if component_ids is None:
            component_ids = interval_component_ids(
                geometry["columns"]
            )

        component_ids = np.asarray(component_ids)

        if component_ids.shape != (len(xyz),):
            raise ValueError("component_ids shape mismatch")

        # QuadrANN branch: exactly one topology/GNN encoding.
        q_features, final_features = self._gnn_features(
            xyz,
            interval_mass,
            component_ids,
        )

        _, inverse_np = np.unique(
            component_ids,
            return_inverse=True,
        )

        inverse = torch.as_tensor(
            inverse_np.copy(),
            device=device,
            dtype=torch.long,
        )

        n_components = int(inverse.max().item()) + 1

        component_context = torch.zeros(
            n_components,
            final_features.shape[-1],
            device=device,
            dtype=dtype,
        )

        component_context = component_context.index_add(
            0,
            inverse,
            final_features,
        )

        component_counts = torch.bincount(
            inverse,
            minlength=n_components,
        ).to(dtype).clamp_min(1)

        component_context = (
            component_context / component_counts[:, None]
        )

        local_component_context = component_context[inverse]

        p_local = pcraft["local_features"].to(dtype)
        p_global = pcraft["global_context"].to(dtype)
        p_global_expanded = p_global.unsqueeze(0).expand(
            len(xyz), -1
        )

        fused = self.fusion(torch.cat((
            q_features,
            final_features,
            p_local,
            p_global_expanded,
            local_component_context,
        ), dim=-1))

        logits = self.weight_head(fused).squeeze(-1)

        # Exactly one mass-prior softmax, once per geometry.
        probabilities = torch.softmax(
            logits.float() + interval_mass.log().float(),
            dim=0,
        ).to(dtype)

        learned_mass = interval_mass.sum() * probabilities

        quadrature = rle_gauss_quadrature(
            geometry,
            order=self.quadrature_order,
        )

        node_xyz = torch.as_tensor(
            quadrature["xyz"].copy(),
            device=device,
            dtype=dtype,
        )

        base_weights = torch.as_tensor(
            quadrature["weights"].copy(),
            device=device,
            dtype=dtype,
        )

        interval_ids = torch.as_tensor(
            quadrature["interval_ids"].copy(),
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
            base_weights / base_interval_mass[interval_ids]
        )

        node_weights = (
            learned_mass[interval_ids] * local_fraction
        )

        if not torch.isfinite(node_weights).all():
            raise FloatingPointError("nonfinite node weights")

        if (node_weights <= 0).any():
            raise FloatingPointError("nonpositive node weights")

        mass_error = (
            node_weights.sum() - interval_mass.sum()
        ).abs()

        if mass_error.item() > 1e-6:
            raise RuntimeError("geometry cache mass conservation failed")

        return SQNOGeometryCache(
            geometry_id=str(
                geometry.get("geometry_sha256", "")
            ),
            node_xyz=node_xyz,
            node_weights=node_weights,
            interval_mass=interval_mass,
            fixed_moments=torch.as_tensor(
                np.asarray(
                    geometry.get("fixed_moments", np.zeros(10)),
                    dtype=np.float64,
                ).copy(),
                device=device,
                dtype=dtype,
            ),
            interval_ids=interval_ids,
            logits=logits,
            probabilities=probabilities,
            pcraft_local_features=p_local,
            pcraft_global_context=p_global,
            quadrann_features=q_features,
            component_context=local_component_context,
            global_scale_weights=pcraft["global_scale_weights"],
            n_intervals=len(interval_mass),
            n_nodes=len(node_xyz),
        )

    def evaluate_queries(
        self,
        cache: SQNOGeometryCache,
        coefficients: torch.Tensor,
        *,
        with_jacobian=True,
    ):
        """Evaluate many queries without rerunning geometry encoders."""

        self.query_evaluation_calls += 1

        if coefficients.ndim == 1:
            coefficients = coefficients.unsqueeze(0)

        if coefficients.ndim != 2 or coefficients.shape[1] != 10:
            raise ValueError("coefficients must be [B,10]")

        # M83 teacher queries are float64. Cast cached geometry
        # differentiably rather than requiring model/label dtypes
        # to be identical.
        query_xyz = cache.node_xyz.to(
            device=coefficients.device,
            dtype=coefficients.dtype,
        )
        query_weights = cache.node_weights.to(
            device=coefficients.device,
            dtype=coefficients.dtype,
        )

        result = self.batched_moment_operator.evaluate(
            query_xyz,
            query_weights,
            coefficients,
            with_jacobian=with_jacobian,
        )

        moments_editable = result["moments"]

        fixed_moments = cache.fixed_moments.to(
            device=moments_editable.device,
            dtype=moments_editable.dtype,
        )

        moments_total = moments_editable + fixed_moments.unsqueeze(0)

        return {
            **result,
            "moments_editable": moments_editable,
            "moments_fixed": fixed_moments,
            "moments_total": moments_total,
            "moments": moments_total,
            "geometry_id": cache.geometry_id,
            "logits": cache.logits,
            "probabilities": cache.probabilities,
            "interval_mass": cache.interval_mass,
            "total_mass_error": (
                cache.node_weights.sum()
                - cache.interval_mass.sum()
            ).abs(),
            "n_intervals": cache.n_intervals,
            "n_nodes": cache.n_nodes,
            "global_scale_weights": cache.global_scale_weights,
        }

    def reset_call_counters(self):
        self.geometry_encode_calls = 0
        self.query_evaluation_calls = 0
