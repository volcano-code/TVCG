"""Named model configurations and factory for Spin Neural Moments V1.

The factory composes existing P-CRAFT, M80 and SQNO modules. It does not
claim that the full candidate is scientifically optimal; it makes the
intended configuration explicit and reproducible.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, replace
import hashlib
import json
from typing import Any, Mapping

import torch

from research.NeuralMoments.models.pcraft_ufno.spin_pino.native_pcraft import (
    PcraftNativeMomentOperator,
    make_config as make_pcraft_config,
)


@dataclass(frozen=True)
class SpinNeuralOperatorConfig:
    name: str
    pcraft_preset: str
    pcraft_grid_size: int | None = None
    native_resolution: int | None = None
    operator_kind: str = "unified_v1"
    v2_token_grid: int = 16
    v2_width: int = 64
    v2_heads: int = 4
    multiscale_grids: tuple[int, ...] | None = None
    pcraft_gno_enabled: bool | None = None
    pcraft_transolver_enabled: bool | None = None
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
    attention_mixing: str = "legacy"
    uq_enabled: bool = True
    event_enabled: bool = True

    def __post_init__(self) -> None:
        if not self.name or not self.pcraft_preset:
            raise ValueError("configuration name and P-CRAFT preset are required")
        if self.fusion_mode not in {"global_scalar", "query_conditioned"}:
            raise ValueError("unsupported fusion_mode")
        if self.attention_mixing not in {"legacy", "sum", "gated"}:
            raise ValueError("unsupported attention_mixing")
        if self.operator_kind not in {"unified_v1","compact_pino_v2"}:
            raise ValueError("unknown operator kind")
        if self.operator_kind=="compact_pino_v2":
            if self.native_resolution!=256 or self.pcraft_gno_enabled or self.pcraft_transolver_enabled or self.uq_enabled or self.event_enabled or self.cross_scale_attention:
                raise ValueError("Compact V2 requires the encoder-only native256 topology")
            if any(isinstance(v,bool) or not isinstance(v,int) or v<1 for v in (self.v2_token_grid,self.v2_width,self.v2_heads)) or self.v2_width%self.v2_heads or self.v2_token_grid>256 or 256%self.v2_token_grid:
                raise ValueError("invalid V2 token/attention dimensions")
        if self.native_resolution is not None and (
            isinstance(self.native_resolution, bool) or not isinstance(self.native_resolution, int)
            or self.native_resolution < 8
        ):
            raise ValueError("native_resolution must be a positive integer >=8")
        if self.pcraft_grid_size is not None and (
            isinstance(self.pcraft_grid_size, bool)
            or not isinstance(self.pcraft_grid_size, int)
            or self.pcraft_grid_size < 8
            or self.pcraft_grid_size % 4
        ):
            raise ValueError("pcraft_grid_size must be >=8 and divisible by four")
        if self.multiscale_grids is not None and (
            not isinstance(self.multiscale_grids, tuple)
            or not self.multiscale_grids
            or any(isinstance(v, bool) or not isinstance(v, int) or v < 8 or v % 4
                   for v in self.multiscale_grids)
            or tuple(sorted(set(self.multiscale_grids))) != self.multiscale_grids
        ):
            raise ValueError("multiscale_grids must be a nonempty sorted tuple of unique valid grids")
        for field in ("pcraft_gno_enabled", "pcraft_transolver_enabled"):
            value = getattr(self, field)
            if value is not None and not isinstance(value, bool):
                raise ValueError(f"{field} must be bool or None")
        if not 0.0 < self.fusion_initial_weight < 1.0:
            raise ValueError("fusion_initial_weight must be in (0,1)")
        if min(self.sqno_hidden_dim, self.sqno_final_hidden_dim,
               self.sqno_num_layers, self.sqno_neighbors, self.query_chunk,
               self.geometry_attention_heads, self.geometry_attention_chunk,
               self.cross_scale_heads, self.cross_scale_max_scales) < 1:
            raise ValueError("model dimensions and chunk sizes must be positive")
        if self.sqno_temperature <= 0:
            raise ValueError("temperature must be positive")
        if (isinstance(self.quadrature_order, bool)
                or not isinstance(self.quadrature_order, int)
                or self.quadrature_order < 2):
            raise ValueError("quadrature_order must be an integer >=2")

    def config_payload(self) -> dict[str, Any]:
        """Preserve legacy metadata exactly when all new controls are unset."""
        values = asdict(self)
        for field, default in (
            ("pcraft_grid_size", None),
            ("native_resolution", None),
            ("operator_kind", "unified_v1"),
            ("v2_token_grid", 16),
            ("v2_width", 64),
            ("v2_heads", 4),
            ("multiscale_grids", None),
            ("pcraft_gno_enabled", None),
            ("pcraft_transolver_enabled", None),
            ("attention_mixing", "legacy"),
        ):
            if values[field] == default:
                del values[field]
        return values

    def fingerprint(self) -> str:
        payload = json.dumps(self.config_payload(), sort_keys=True, separators=(",", ":"))
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


# All G/T arms start from the exact same native preset. Switching presets
# would also change inactive-module defaults such as slice heads and topology
# filtering, making the experiment contract less precise.
_ABLATION_BASE = replace(
    CONFIGS["full_v1_candidate"],
    name="ablation_g1_t1",
    interaction_variant="independent",
    attention_mixing="gated",
    pcraft_gno_enabled=True,
    pcraft_transolver_enabled=True,
)
for _gno, _transolver in ((False, False), (True, False), (False, True), (True, True)):
    _name = f"ablation_g{int(_gno)}_t{int(_transolver)}"
    CONFIGS[_name] = replace(
        _ABLATION_BASE, name=_name,
        pcraft_gno_enabled=_gno, pcraft_transolver_enabled=_transolver,
    )
CONFIGS["ablation_no_cross_scale"] = replace(
    _ABLATION_BASE, name="ablation_no_cross_scale", cross_scale_attention=False,
)
CONFIGS["ablation_two_scales"] = replace(
    _ABLATION_BASE, name="ablation_two_scales", multiscale_grids=(64, 128),
)
CONFIGS["ablation_one_scale"] = replace(
    _ABLATION_BASE, name="ablation_one_scale", multiscale_grids=(64,),
)
# A combined simplification candidate, not a single-factor ablation or a
# performance-validated replacement for the existing full/reference models.
CONFIGS["compact_v1_candidate"] = replace(
    _ABLATION_BASE,
    name="compact_v1_candidate",
    pcraft_gno_enabled=False,
    pcraft_transolver_enabled=False,
    multiscale_grids=(64, 128),
    cross_scale_attention=False,
    attention_mixing="sum",
    uq_enabled=False,
    event_enabled=False,
)

# Explicit 256-native candidates. Historical configurations are unchanged.
# The common reference is a weight initializer, not a 256 geometry encoder.
for _old in ("ablation_g1_t1", "ablation_g0_t0", "ablation_g1_t0", "ablation_g0_t1",
             "ablation_no_cross_scale", "compact_v1_candidate"):
    _name = _old + "_256"
    CONFIGS[_name] = replace(CONFIGS[_old], name=_name, native_resolution=256)
CONFIGS["compact_lite_256"] = replace(
    CONFIGS["compact_v1_candidate_256"], name="compact_lite_256",
    multiscale_grids=(32, 64),
)
CONFIGS["compact_single_256"] = replace(
    CONFIGS["compact_v1_candidate_256"], name="compact_single_256",
    multiscale_grids=(64,),
)

# Single-response V2; matched P-CRAFT width and grids for a controlled first comparison.
CONFIGS["compact_pino_v2_256"] = replace(
    CONFIGS["compact_lite_256"], name="compact_pino_v2_256", operator_kind="compact_pino_v2",
)
CONFIGS["compact_pino_v2_tokens8_256"] = replace(
    CONFIGS["compact_pino_v2_256"], name="compact_pino_v2_tokens8_256", v2_token_grid=8,
)


def get_config(name: str, **overrides: Any) -> SpinNeuralOperatorConfig:
    """Return a named immutable config, optionally with explicit overrides."""
    if name not in CONFIGS:
        raise KeyError(f"unknown Spin Neural Operator config: {name}")
    values = asdict(CONFIGS[name])
    values.update(overrides)
    # JSON configuration files express tuples as lists. Canonicalize before
    # validation/fingerprinting so API and file-based runs agree.
    if isinstance(values.get("multiscale_grids"), list):
        values["multiscale_grids"] = tuple(values["multiscale_grids"])
    return SpinNeuralOperatorConfig(**values)


def _make_native_config(cfg: SpinNeuralOperatorConfig):
    """Resolve runtime geometry overrides identically for build and reporting."""
    overrides: dict[str, Any] = {
        "uq_enabled": cfg.uq_enabled,
        "event_validity_head_enabled": cfg.event_enabled,
    }
    if cfg.native_resolution is not None:
        overrides["min_native_resolution"] = cfg.native_resolution
    if cfg.pcraft_grid_size is not None:
        overrides["grid_size"] = cfg.pcraft_grid_size
    if cfg.pcraft_gno_enabled is not None:
        overrides["gno_enabled"] = cfg.pcraft_gno_enabled
    if cfg.pcraft_transolver_enabled is not None:
        overrides["transolver_enabled"] = cfg.pcraft_transolver_enabled
    if cfg.multiscale_grids is not None:
        if len(cfg.multiscale_grids) == 1:
            grid = cfg.multiscale_grids[0]
            if cfg.pcraft_grid_size is not None and cfg.pcraft_grid_size != grid:
                raise ValueError("single-scale grid conflicts with pcraft_grid_size")
            # Native config retains a valid, unused >=2-entry scale tuple.
            # The effective input consists only of grid_size in this mode.
            overrides.update(grid_size=grid, multiscale_enabled=False)
        else:
            overrides.update(multiscale_enabled=True, multiscale_grids=cfg.multiscale_grids)
    native = make_pcraft_config(cfg.pcraft_preset, **overrides)
    if cfg.native_resolution is not None:
        grids = native.multiscale_grids if native.multiscale_enabled else (native.grid_size,)
        if any(cfg.native_resolution % g for g in grids):
            raise ValueError("all context grids must divide exact native_resolution")
    return native


def build_model(
    name: str = "full_v1_candidate",
    *,
    device: str | torch.device | None = None,
    config_overrides: Mapping[str, Any] | None = None,
) -> torch.nn.Module:
    """Construct a complete unified model from a named V1 configuration."""
    cfg = get_config(name, **dict(config_overrides or {}))
    pcraft_cfg = _make_native_config(cfg)
    if cfg.operator_kind=="compact_pino_v2":
        from .compact_pino_v2 import CompactPINOv2
        encoder=PcraftNativeMomentOperator(pcraft_cfg,encoder_only=True)
        model=CompactPINOv2(encoder,native_resolution=cfg.native_resolution,token_grid=cfg.v2_token_grid,
            width=cfg.v2_width,heads=cfg.v2_heads,query_chunk=cfg.query_chunk,
            architecture_metadata={"named_config":cfg.name,"config_fingerprint":cfg.fingerprint(),"config":cfg.config_payload()})
        return model.to(device) if device is not None else model
    # SQNO's optional graph extensions are needed only by the unified V1 arm.
    # Reading configurations or constructing a compact operator must not load
    # those unrelated native extensions.
    from .spin_quadrann_v12 import SpinQuadrANNV12
    from .unified_m80_sqno import UnifiedM80SQNO

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
            "config": cfg.config_payload(),
        },
        cross_attention_enabled=cfg.geometry_cross_attention,
        cross_attention_heads=cfg.geometry_attention_heads,
        cross_attention_chunk=cfg.geometry_attention_chunk,
        cross_attention_output_scale=cfg.geometry_attention_output_scale,
        cross_scale_attention_enabled=cfg.cross_scale_attention,
        cross_scale_heads=cfg.cross_scale_heads,
        cross_scale_max_scales=cfg.cross_scale_max_scales,
        interaction_variant=cfg.interaction_variant,
        attention_mixing=cfg.attention_mixing,
        fusion_mode=cfg.fusion_mode,
        query_fusion_hidden=cfg.query_fusion_hidden,
    )
    if device is not None:
        model = model.to(device)
    return model


def architecture_manifest(
    name: str, *, config_overrides: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Report the resolved run configuration without allocating model weights."""
    cfg = get_config(name, **dict(config_overrides or {}))
    pcraft = _make_native_config(cfg)
    effective_grids = pcraft.multiscale_grids if pcraft.multiscale_enabled else (pcraft.grid_size,)
    if cfg.operator_kind=="compact_pino_v2":
        return {"schema":"tvcg.compact-pino.v2","name":cfg.name,"config":cfg.config_payload(),
                "config_fingerprint":cfg.fingerprint(),"pcraft_config":asdict(pcraft),
                "effective_geometry":{"pcraft_grid_size":pcraft.grid_size,"grids":list(effective_grids),
                    "representation":"native_rle","required_native_resolution":cfg.native_resolution,
                    "token_grid":cfg.v2_token_grid,"max_physical_tokens":cfg.v2_token_grid**3},
                "response":{"width":cfg.v2_width,"heads":cfg.v2_heads,"query_chunk":cfg.query_chunk,
                    "kind":"bounded_physical_token_integral","query_domain":"positive_quadric_on_editable_domain"},
                "modules":{"shared_pcraft":True,"m80":False,"sqno":False,"sqno_gnn":False,
                    "gno":False,"transolver_physics_slicing":False,"cross_scale_attention":False,
                    "query_conditioned_fusion":False,"geometry_physics_cross_attention":True,
                    "single_physical_response":True,"moment_jacobian_uq":False,"event_validity":False},
                "scientific_status":"candidate_configuration_not_performance_validated"}
    effective_mixing = cfg.attention_mixing
    if effective_mixing == "legacy":
        effective_mixing = "gated" if cfg.interaction_variant == "parallel_gno_attention" else "sum"
    return {
        "schema": "tvcg.spin-neural-moments.v1",
        "name": cfg.name,
        "config": cfg.config_payload(),
        "config_fingerprint": cfg.fingerprint(),
        "pcraft_config": asdict(pcraft),
        "effective_geometry": {
            "pcraft_grid_size": pcraft.grid_size,
            "multiscale_enabled": pcraft.multiscale_enabled,
            "grids": list(effective_grids),
            "representation": "native_rle",
            "required_native_resolution": cfg.native_resolution,
            "quadrature_order": cfg.quadrature_order,
        },
        "attention_mixing": cfg.attention_mixing,
        "effective_attention_mixing": effective_mixing,
        "modules": {
            "shared_pcraft": True,
            "m80": True,
            "sqno": True,
            "sqno_gnn": True,
            "multiscale_pcraft": pcraft.multiscale_enabled,
            "global_scale_gate": pcraft.multiscale_enabled,
            "local_scale_gate": pcraft.multiscale_enabled,
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
