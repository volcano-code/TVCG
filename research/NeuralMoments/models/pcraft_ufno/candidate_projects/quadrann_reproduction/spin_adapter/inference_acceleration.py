"""Optional exact inference caches with an explicit persistent byte budget.

These caches are never used during training. They preserve the existing
quadrature nodes and global mass-weighted attention, rather than approximating
or dropping samples. The budget covers additional persistent tensor storage,
not total allocator peak or temporary workspace.
"""
from dataclasses import dataclass
from typing import Any

import torch

from .spin_moment_operator import raw_moment_basis


def _signature(tensors):
    if any(t.is_inference() for t in tensors):
        raise ValueError("versioned acceleration requires no_grad, not inference_mode tensors")
    return tuple((id(t), t._version, tuple(t.shape), t.dtype, t.device) for t in tensors)


def _sources(cache):
    return (cache.shared_cache.interval_features, cache.shared_cache.interval_mass,
            cache.sqno_cache.node_xyz, cache.sqno_cache.node_weights)


@dataclass(frozen=True)
class InferenceAcceleration:
    source_signature: tuple
    prepared_signature: tuple
    attention_kv: tuple | None
    prepared_query: tuple | None
    extra_bytes: int

    def validate(self, cache):
        prepared = (self.attention_kv or ()) + (self.prepared_query or ())
        if self.source_signature != _signature(_sources(cache)) or self.prepared_signature != _signature(prepared):
            raise ValueError("stale or mutated inference acceleration cache")


@torch.no_grad()
def prepare_acceleration(model, cache, max_extra_bytes: int, *, attention=True, quadrature=True):
    if isinstance(max_extra_bytes,bool) or not isinstance(max_extra_bytes,int) or max_extra_bytes<0:
        raise ValueError("max_extra_bytes must be a nonnegative integer")
    if model.training or not cache.inference_only:
        raise ValueError("acceleration requires eval() and inference-only geometry cache")
    if cache.owner!=id(model) or cache.versions!=model._versions():
        raise ValueError("foreign or stale unified geometry cache")
    if getattr(cache,"acceleration",None) is not None:
        raise ValueError("prepare acceleration from the original unaccelerated cache")
    signature = _signature(_sources(cache))
    f=cache.shared_cache.interval_features
    xyz=cache.sqno_cache.node_xyz
    weights=cache.sqno_cache.node_weights
    attention_bytes = 2*f.numel()*f.element_size() if model.geometry_physics_attention is not None else 0
    basis_bytes = len(xyz)*10*8
    cast_bytes = (xyz.numel()*8 if xyz.dtype!=torch.float64 else 0)+(weights.numel()*8 if weights.dtype!=torch.float64 else 0)
    quadrature_bytes = basis_bytes+cast_bytes
    used=0
    kv=None
    prepared=None
    if attention and attention_bytes and used+attention_bytes<=max_extra_bytes:
        kv=model.geometry_physics_attention.prepare_projected_kv(f)
        used+=attention_bytes
    if quadrature and used+quadrature_bytes<=max_extra_bytes:
        qxyz=xyz.to(dtype=torch.float64)
        qw=weights.to(dtype=torch.float64)
        basis=qxyz.new_empty((len(qxyz),10))
        chunk=model.sqno.batched_moment_operator.chunk_nodes
        for i in range(0,len(qxyz),chunk):
            basis[i:i+chunk]=raw_moment_basis(qxyz[i:i+chunk])
        prepared=(qxyz,qw,basis)
        used+=quadrature_bytes
    prepared_signature=_signature((kv or ())+(prepared or ()))
    state=InferenceAcceleration(signature,prepared_signature,kv,prepared,used)
    report={"max_extra_bytes":max_extra_bytes,"extra_bytes":used,
            "attention_cached":kv is not None,"quadrature_cached":prepared is not None,
            "attention_required_bytes":attention_bytes,"quadrature_required_bytes":quadrature_bytes,
            "budget_scope":"additional_persistent_tensors_not_peak","physical_nodes_changed":False}
    return state,report
