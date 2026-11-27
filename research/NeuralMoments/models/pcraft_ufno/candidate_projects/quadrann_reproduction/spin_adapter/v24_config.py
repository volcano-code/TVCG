"""Explicit experimental V2.4 geometry/operator contract."""
from __future__ import annotations
from dataclasses import asdict, dataclass
import hashlib, json

@dataclass(frozen=True)
class CompactPINOv24Config:
    native_material_resolution: int = 256
    pcraft_context_grids: tuple[int, ...] = (64, 128)
    token_grid: int = 16
    geometry_width: int = 192
    cut_token_count: int = 16
    cut_attention_heads: int = 4
    cut_attention_layers: int = 1
    cut_attention_scope: str = "active_band_only"
    response_channels: int = 2
    moment_dim: int = 10
    jacobian_from_same_moment_graph: bool = True
    exact_native_acceptance: bool = True

    def __post_init__(self):
        if self.native_material_resolution != 256:
            raise ValueError("V2.4 keeps native 256^3 material for this experiment")
        if self.pcraft_context_grids != (64, 128):
            raise ValueError("first V2.4 scale candidate is exactly (64,128)")
        if any(self.native_material_resolution % g for g in self.pcraft_context_grids):
            raise ValueError("context grids must divide native material resolution")
        if self.token_grid < 1 or self.native_material_resolution % self.token_grid:
            raise ValueError("token_grid must divide native material resolution")
        if self.geometry_width % self.cut_attention_heads:
            raise ValueError("geometry_width must divide attention heads")
        if self.cut_token_count < 1 or self.cut_attention_layers != 1:
            raise ValueError("first candidate uses positive cut tokens and one layer")
        if self.cut_attention_scope != "active_band_only":
            raise ValueError("global all-token attention is excluded")
        if self.moment_dim != 10 or not self.jacobian_from_same_moment_graph or not self.exact_native_acceptance:
            raise ValueError("physical output/acceptance contract cannot be weakened")

    def fingerprint(self) -> str:
        payload = json.dumps(asdict(self), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(payload.encode()).hexdigest()
