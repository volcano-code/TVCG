"""SQNO v1.2: geometry-query decoupled model.

Geometry is encoded once, then reused for batched coefficient queries.
The geometry cache deliberately retains its autograd graph during training.
"""
from __future__ import annotations

from dataclasses import dataclass
from itertools import chain
import numpy as np
import torch
from torch import nn
from research.NeuralMoments.models.pcraft_ufno.spin_pino.native_pcraft import (
    NativeFeatureSource, PreparedNativeGeometry, build_local_neighbor_graph, checkpoint_stage, parameter_versions,
)
from .memory_profile import profile_stage, profile_count, profile_tensors, profile_adjoint

from .spin_quadrann_v11 import SpinQuadrANNV11
from .batched_moment_operator import BatchedSpinMomentOperator
from .prepared_quadrature import prepared_quadrature


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
    owner: int
    versions: tuple
    native_feature_source: NativeFeatureSource | None
    inference_only: bool


class SpinQuadrANNV12(SpinQuadrANNV11):
    """Geometry encoder plus batched query evaluator."""

    def __init__(self, *args, query_chunk=16, share_pcraft_encoder=False,
                 pcraft_width=None, fusion_chunk_size=1024, **kwargs):
        super().__init__(*args, pcraft_width=pcraft_width, **kwargs)
        self.external_pcraft_required = pcraft_width is not None
        self.query_chunk = int(query_chunk)
        self.fusion_chunk_size = int(fusion_chunk_size)
        if self.fusion_chunk_size < 1:
            raise ValueError("fusion_chunk_size must be positive")

        self.batched_moment_operator = BatchedSpinMomentOperator(
            temperature=self.moment_operator.temperature,
            chunk_nodes=min(self.moment_operator.chunk_nodes, 8192),
            chunk_queries=int(query_chunk),
        )

        self.share_pcraft_encoder=bool(share_pcraft_encoder)
        if self.external_pcraft_required:
            self.share_pcraft_encoder = True
        elif self.share_pcraft_encoder:
            self.pcraft.requires_grad_(False)
        self.geometry_encode_calls = 0
        self.query_evaluation_calls = 0

    def encode_geometry(self, geometry, component_ids=None, *, pcraft_features=None):
        """Encode one geometry and retain a differentiable cache."""

        self.geometry_encode_calls += 1

        geometry = PreparedNativeGeometry.prepare(geometry,borrow=True)

        parameter = next(self.parameters())
        device = parameter.device
        dtype = parameter.dtype

        # P-CRAFT may be supplied by the shared UG-PINO encoder. In that
        # mode this SQNO branch performs only its topology/GNN and query work.
        native_source=None
        if pcraft_features is None:
            if self.share_pcraft_encoder:
                raise ValueError("shared P-CRAFT mode requires externally encoded pcraft_features")
            pcraft = self.pcraft(geometry)
        else:
            pcraft = dict(pcraft_features)
            native_source=pcraft.get("native_feature_source")
            if native_source is not None:
                if not isinstance(native_source,NativeFeatureSource):
                    raise ValueError("invalid shared P-CRAFT feature source")
                native_source.validate()
            required=("local_features","global_context","global_scale_weights","interval_mass")
            missing=[key for key in required if key not in pcraft]
            if missing:
                raise ValueError("shared P-CRAFT feature contract missing: "+", ".join(missing))
            expected_width=self.pcraft_width
            if pcraft["local_features"].ndim!=2 or pcraft["local_features"].shape != (len(geometry["xyz"]),expected_width):
                raise ValueError(
                    "shared P-CRAFT local feature shape mismatch; construct SQNO with a preset "
                    "whose interval_width matches the UG-PINO encoder (e.g. preset='xl_ms' for width 192)"
                )
            if pcraft["global_context"].shape!=(expected_width,):
                raise ValueError("shared P-CRAFT global context width mismatch")
            if pcraft["interval_mass"].shape!=(len(geometry["xyz"]),):
                raise ValueError("shared P-CRAFT interval mass shape mismatch")
            if pcraft["global_scale_weights"].ndim not in (1,2):
                raise ValueError("shared P-CRAFT global scale weights must be rank 1 or 2")
            supplied_id=str(pcraft.get("geometry_id",""))
            geometry_id=str(geometry.get("geometry_sha256",""))
            if not supplied_id or not geometry_id or supplied_id != geometry_id:
                raise ValueError("shared P-CRAFT geometry identity missing or mismatched")
            for key in ("local_features","global_context","global_scale_weights","interval_mass"):
                if pcraft[key].device != device:
                    raise ValueError(f"shared P-CRAFT {key} device mismatch")
                if not torch.isfinite(pcraft[key]).all():
                    raise FloatingPointError(f"shared P-CRAFT {key} contains nonfinite values")
            if (pcraft["interval_mass"] <= 0).any():
                raise ValueError("shared P-CRAFT interval masses must be positive")
            geometry_mass=torch.as_tensor(geometry["mass"].sum(),device=device,dtype=torch.float64)
            if not torch.allclose(pcraft["interval_mass"].sum().to(torch.float64),geometry_mass,rtol=1e-6,atol=1e-9):
                raise ValueError("shared P-CRAFT interval mass does not match native geometry")

        xyz = torch.as_tensor(
            np.asarray(geometry["xyz"]).copy(),
            device=device,
            dtype=dtype,
        )

        interval_mass = pcraft["interval_mass"].to(
            device=device,
            dtype=dtype,
        )

        supplied_components = component_ids is not None
        if component_ids is None:
            component_ids = geometry.components()

        component_ids = np.asarray(component_ids)

        if component_ids.shape != (len(xyz),):
            raise ValueError("component_ids shape mismatch")

        # QuadrANN branch: exactly one topology/GNN encoding.
        with profile_stage("sqno_gnn", native_intervals=len(xyz)):
            xyz_np=xyz.detach().cpu().numpy()
            graph=(None if supplied_components else geometry.derived(
                ("knn",self.gnn.neighbors,xyz_np.dtype.str,True),
                lambda: build_local_neighbor_graph(xyz_np,self.gnn.neighbors,component_ids)))
            q_feature_layers, final_features = self._gnn_features(
                xyz,interval_mass,component_ids,return_layer_features=True,graph=graph)
            profile_tensors(final_gnn_features=final_features)
            profile_adjoint("sqno_gnn",outputs=q_feature_layers,inputs=(xyz,),
                            parameters=self.gnn.parameters())

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

        p_local = pcraft["local_features"].to(dtype=dtype)
        p_global = pcraft["global_context"].to(dtype=dtype)

        # The concatenated per-interval fusion tensor is several GiB at
        # million-interval resolution. Evaluate the exact same pointwise
        # fusion MLP in chunks and checkpoint each chunk during training.
        with profile_stage("sqno_fusion", native_intervals=len(xyz)):
            logit_chunks = []
            n_gnn_layers = len(q_feature_layers)
            split_layers = [layer.split(self.fusion_chunk_size) for layer in q_feature_layers]
            split_local = p_local.split(self.fusion_chunk_size)
            for i,start in enumerate(range(0, len(xyz), self.fusion_chunk_size)):
                stop = min(start + self.fusion_chunk_size, len(xyz))
                layer_chunks = [layer[i] for layer in split_layers]
                p_chunk = split_local[i]
                inverse_chunk = inverse[start:stop]

                def fusion_chunk_fn(*inputs,
                                    n_layers=n_gnn_layers,
                                    fusion_module=self.fusion,
                                    weight_module=self.weight_head,
                                    global_context=p_global,
                                    component_summary=component_context):
                    profile_count(fusion_chunks=1)
                    layer_inputs = inputs[:n_layers]
                    local_p = inputs[n_layers]
                    inv = inputs[n_layers + 1]
                    qf = torch.cat(layer_inputs, dim=-1)
                    ff = layer_inputs[-1]
                    component_local = component_summary[inv]
                    global_local = global_context.unsqueeze(0).expand(len(qf), -1)
                    fused_chunk = fusion_module(torch.cat(
                        (qf, ff, local_p, global_local, component_local), dim=-1
                    ))
                    return weight_module(fused_chunk).squeeze(-1)

                checkpoint_inputs = (*layer_chunks, p_chunk, inverse_chunk)
                if self.training and torch.is_grad_enabled():
                    chunk_logits = checkpoint_stage("sqno_fusion_recompute",fusion_chunk_fn,*checkpoint_inputs)
                else:
                    chunk_logits = fusion_chunk_fn(*checkpoint_inputs)
                logit_chunks.append(chunk_logits)

            logits = torch.cat(logit_chunks, dim=0)
            profile_adjoint("sqno_fusion",outputs=logits,
                            inputs=(*q_feature_layers,p_local,p_global,component_context),
                            parameters=chain(self.fusion.parameters(),self.weight_head.parameters()))

        # Exactly one mass-prior softmax, once per geometry.
        probabilities = torch.softmax(
            logits.float() + interval_mass.log().float(),
            dim=0,
        ).to(dtype)

        learned_mass = interval_mass.sum() * probabilities

        with profile_stage("sqno_quadrature", native_intervals=len(xyz)):
            storage_dtype=np.float64 if dtype==torch.float64 else np.float32
            node_array, weight_array, mass_array=geometry.derived(
                ("quadrature",self.quadrature_order,np.dtype(storage_dtype).str),
                lambda: prepared_quadrature(geometry,self.quadrature_order,storage_dtype))
            node_xyz=torch.as_tensor(node_array,device=device,dtype=dtype)
            base_weights=torch.as_tensor(weight_array,device=device,dtype=dtype)
            base_interval_mass=torch.as_tensor(mass_array,device=device,dtype=dtype)
            interval_ids=torch.arange(len(xyz),device=device).repeat_interleave(self.quadrature_order**3)
            profile_tensors(quadrature_coordinates=node_xyz)

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

        inference_only=bool(native_source and native_source.inference_only)
        if not torch.is_grad_enabled():
            trainable=chain(self.gnn.parameters(),self.fusion.parameters(),self.weight_head.parameters(),
                            self.pcraft.operator.geometry_parameters() if self.pcraft is not None else ())
            inference_only=inference_only or p_local.requires_grad or p_global.requires_grad or any(
                p.requires_grad for p in trainable)
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
            quadrann_features=final_features,
            component_context=component_context,
            global_scale_weights=pcraft["global_scale_weights"],
            n_intervals=len(interval_mass),
            n_nodes=len(node_xyz),
            owner=id(self),
            versions=parameter_versions(self),
            native_feature_source=native_source,
            inference_only=inference_only,
        )

    def evaluate_queries(
        self,
        cache: SQNOGeometryCache,
        coefficients: torch.Tensor,
        *,
        with_jacobian=True,
        prepared_query=None,
    ):
        """Evaluate many queries without rerunning geometry encoders."""

        if cache.owner!=id(self) or cache.versions!=parameter_versions(self):
            raise ValueError("foreign or stale SQNO geometry cache")
        if cache.native_feature_source is not None:
            cache.native_feature_source.validate()
        if cache.inference_only and self.training and torch.is_grad_enabled():
            raise ValueError("inference-only SQNO geometry cache cannot be used for training")
        self.query_evaluation_calls += 1

        if coefficients.ndim == 1:
            coefficients = coefficients.unsqueeze(0)

        if coefficients.ndim != 2 or coefficients.shape[1] != 10:
            raise ValueError("coefficients must be [B,10]")

        # M83 teacher queries are float64. Cast cached geometry
        # differentiably rather than requiring model/label dtypes
        # to be identical.
        if prepared_query is None:
            query_xyz = cache.node_xyz.to(
                device=coefficients.device, dtype=coefficients.dtype)
            query_weights = cache.node_weights.to(
                device=coefficients.device, dtype=coefficients.dtype)
            basis = None
        else:
            if self.training or torch.is_grad_enabled() or not cache.inference_only:
                raise ValueError("prepared quadrature requires inference-only cache and no_grad")
            query_xyz, query_weights, basis = prepared_query
            if query_xyz.shape!=cache.node_xyz.shape or query_weights.shape!=cache.node_weights.shape:
                raise ValueError("prepared quadrature shape mismatch")

        result = self.batched_moment_operator.evaluate(
            query_xyz,
            query_weights,
            coefficients,
            with_jacobian=with_jacobian, prepared_basis=basis,
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
