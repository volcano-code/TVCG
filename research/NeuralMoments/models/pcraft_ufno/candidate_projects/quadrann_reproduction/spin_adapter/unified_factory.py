"""Named model configurations and factory for Spin Neural Moments V1.

The factory composes existing P-CRAFT, M80 and SQNO modules. It does not
claim that the full candidate is scientifically optimal; it makes the
intended configuration explicit and reproducible.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from typing import Any, Mapping

import torch

from research.NeuralMoments.models.pcraft_ufno.spin_pino.native_pcraft import (
    PcraftNativeMomentOperator,
    make_config as make_pcraft_config,
)
from .spin_quadrann_v12 import SpinQuadrANNV12
from .unified_m80_sqno import UnifiedM80SQNO


@dataclass(frozen=True)
class SpinNeuralOperatorConfig:
    name: str
    pcraft_preset: str
    sqno_hidden_dim: int = 32
    sqno_final_hidden_dim: int = 128
    sqno_num_layers: int = 3
    sqno_neighbors: int = 16
    sqno_temperature: float = 0.04
    quadrature_order: int = 2
    sqno_chunk_intervals: int = 512
    sqno_chunk_nodes: int = 65536
    query_chunk: int = 16
    fusion_mode: str = "global_scalar"
    fusion_initial_weight: float = 0.5
    query_fusion_hidden: int = 32
    geometry_cross_attention: bool = False
    geometry_attention_heads: int = 2
    geometry_attention_chunk: int = 1024
    geometry_attention_output_scale: float = 0.05
    cross_scale_attention: bool = False
    cross_scale_heads: int = 2
    cross_scale_max_scales: int = 16
    interaction_variant: str = "independent"
    uq_enabled: bool = True
    event_enabled: bool = True

    def __post_init__(self) -> None:
        if not self.name or not self.pcraft_preset:
            raise ValueError("configuration name and P-CRAFT preset are required")
        if self.fusion_mode not in {"global_scalar", "query_conditioned"}:
            raise ValueError("unsupported fusion_mode")
        if not 0.0 < self.fusion_initial_weight < 1.0:
            raise ValueError("fusion_initial_weight must be in (0,1)")
        if min(self.sqno_hidden_dim, self.sqno_final_hidden_dim,
               self.sqno_num_layers, self.sqno_neighbors, self.query_chunk,
               self.geometry_attention_heads, self.geometry_attention_chunk,
               self.cross_scale_heads, self.cross_scale_max_scales) < 1:
            raise ValueError("model dimensions and chunk sizes must be positive")
        if self.sqno_temperature <= 0 or self.quadrature_order < 1:
            raise ValueError("temperature and quadrature order must be positive")

    def fingerprint(self) -> str:
        payload = json.dumps(asdict(self), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()


CONFIGS: dict[str, SpinNeuralOperatorConfig] = {
    "reference_v1": SpinNeuralOperatorConfig(
        name="reference_v1",
        pcraft_preset="xl_ms",
        fusion_mode="global_scalar",
        geometry_cross_attention=False,
        cross_scale_attention=False,
        interaction_variant="independent",
        uq_enabled=False,
        event_enabled=False,
    ),
    "full_v1_candidate": SpinNeuralOperatorConfig(
        name="full_v1_candidate",
        pcraft_preset="xl_ms_gno_transolver",
        fusion_mode="query_conditioned",
        geometry_cross_attention=True,
        cross_scale_attention=True,
        interaction_variant="parallel_gno_attention",
        uq_enabled=True,
        event_enabled=True,
    ),
}


def get_config(name: str, **overrides: Any) -> SpinNeuralOperatorConfig:
    """Return a named immutable config, optionally with explicit overrides."""
    if name not in CONFIGS:
        raise KeyError(f"unknown Spin Neural Operator config: {name}")
    values = asdict(CONFIGS[name])
    values.update(overrides)
    return SpinNeuralOperatorConfig(**values)


def build_model(
    name: str = "full_v1_candidate",
    *,
    device: str | torch.device | None = None,
    config_overrides: Mapping[str, Any] | None = None,
) -> UnifiedM80SQNO:
    """Construct a complete unified model from a named V1 configuration."""
    cfg = get_config(name, **dict(config_overrides or {}))
    pcraft_cfg = make_pcraft_config(
        cfg.pcraft_preset,
        uq_enabled=cfg.uq_enabled,
        event_validity_head_enabled=cfg.event_enabled,
    )
    encoder = PcraftNativeMomentOperator(pcraft_cfg)
    sqno = SpinQuadrANNV12(
        preset=cfg.pcraft_preset,
        hidden_dim=cfg.sqno_hidden_dim,
        final_hidden_dim=cfg.sqno_final_hidden_dim,
        num_layers=cfg.sqno_num_layers,
        neighbors=cfg.sqno_neighbors,
        temperature=cfg.sqno_temperature,
        quadrature_order=cfg.quadrature_order,
        chunk_intervals=cfg.sqno_chunk_intervals,
        chunk_nodes=cfg.sqno_chunk_nodes,
        query_chunk=cfg.query_chunk,
        pcraft_width=pcraft_cfg.interval_width,
    )
    model = UnifiedM80SQNO(
        encoder,
        sqno,
        fusion_initial_weight=cfg.fusion_initial_weight,
        architecture_metadata={
            "named_config": cfg.name,
            "config_fingerprint": cfg.fingerprint(),
            "config": asdict(cfg),
        },
        cross_attention_enabled=cfg.geometry_cross_attention,
        cross_attention_heads=cfg.geometry_attention_heads,
        cross_attention_chunk=cfg.geometry_attention_chunk,
        cross_attention_output_scale=cfg.geometry_attention_output_scale,
        cross_scale_attention_enabled=cfg.cross_scale_attention,
        cross_scale_heads=cfg.cross_scale_heads,
        cross_scale_max_scales=cfg.cross_scale_max_scales,
        interaction_variant=cfg.interaction_variant,
        fusion_mode=cfg.fusion_mode,
        query_fusion_hidden=cfg.query_fusion_hidden,
    )
    if device is not None:
        model = model.to(device)
    return model


def architecture_manifest(name: str) -> dict[str, Any]:
    """Machine-readable intended configuration without allocating model weights."""
    cfg = get_config(name)
    pcraft = make_pcraft_config(cfg.pcraft_preset,
        uq_enabled=cfg.uq_enabled,
        event_validity_head_enabled=cfg.event_enabled)
    return {
        "schema": "tvcg.spin-neural-moments.v1",
        "name": cfg.name,
        "config": asdict(cfg),
        "config_fingerprint": cfg.fingerprint(),
        "pcraft_config": asdict(pcraft),
        "modules": {
            "shared_pcraft": True,
            "m80": True,
            "sqno": True,
            "gno": pcraft.gno_enabled,
            "transolver_physics_slicing": pcraft.transolver_enabled,
            "geometry_physics_cross_attention": cfg.geometry_cross_attention,
            "cross_scale_attention": cfg.cross_scale_attention,
            "query_conditioned_fusion": cfg.fusion_mode == "query_conditioned",
            "moment_jacobian_uq": pcraft.uq_enabled,
            "event_validity": pcraft.event_validity_head_enabled,
        },
        "scientific_status": "candidate_configuration_not_performance_validated",
    }


__all__ = [
    "SpinNeuralOperatorConfig",
    "CONFIGS",
    "get_config",
    "build_model",
    "architecture_manifest",
]
