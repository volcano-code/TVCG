"""Reproducible fresh-run initialization for structural ablations.

Optional modules consume different amounts of random state during construction.
Using the same seed alone therefore does not align the remaining parameters.
This module copies a newly constructed CPU reference by state name, shape and
dtype. It never reads a checkpoint or copies an optimizer.
"""
from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from typing import Any

import torch
from torch import Tensor, nn

from .unified_factory import build_model


def state_fingerprint(state: Mapping[str, Tensor]) -> str:
    """Hash names, shapes, dtypes and exact tensor bytes in a stable order."""
    digest = hashlib.sha256()
    for name in sorted(state):
        value = state[name].detach().cpu().contiguous()
        header = json.dumps([name, str(value.dtype), list(value.shape)],
                            separators=(",", ":")).encode("utf-8")
        raw = value.reshape(-1).view(torch.uint8).numpy().tobytes()
        for part in (header, raw):
            digest.update(len(part).to_bytes(8, "big"))
            digest.update(part)
    return digest.hexdigest()


def initialize_from_reference(
    target: nn.Module,
    *,
    reference_config_name: str = "full_v1_candidate",
    seed: int = 830251,
    require_complete: bool = True,
) -> dict[str, Any]:
    """Copy compatible random reference state without consuming caller RNG.

    Complete coverage is the default for the supplied ablations, whose state
    dictionaries are subsets of Full V1. ``require_complete=False`` permits
    explicitly reported target-only tensors to retain their constructor state.
    All compatibility checks precede copying; a rejected complete copy leaves
    the target unchanged. Special initialization (including zero output heads)
    comes from the reference constructor, rather than generic reinitialization.
    """
    if isinstance(seed, bool) or not isinstance(seed, int) or not 0 <= seed < 2**63:
        raise ValueError("reference seed must be an integer in [0, 2**63)")
    if not isinstance(require_complete, bool):
        raise TypeError("require_complete must be bool")
    if not isinstance(reference_config_name, str) or not reference_config_name:
        raise ValueError("reference_config_name must be nonempty")

    # Seed only the CPU generator. torch.manual_seed would also alter CUDA RNG
    # states, which are intentionally outside this CPU-only reference build.
    with torch.random.fork_rng(devices=[]):
        generator = torch.Generator(device="cpu").manual_seed(seed)
        torch.set_rng_state(generator.get_state())
        reference = build_model(reference_config_name, device="cpu")

    source = reference.state_dict()
    destination = target.state_dict()
    copied = {}
    missing = []
    mismatched = []
    for name, value in destination.items():
        if name not in source:
            missing.append(name)
        elif value.shape != source[name].shape or value.dtype != source[name].dtype:
            mismatched.append({
                "key": name,
                "target_shape": list(value.shape),
                "reference_shape": list(source[name].shape),
                "target_dtype": str(value.dtype),
                "reference_dtype": str(source[name].dtype),
            })
        else:
            copied[name] = source[name]
    if not copied:
        raise ValueError("reference and target have no compatible state tensors")
    if require_complete and (missing or mismatched):
        details = json.dumps({"missing_keys": missing, "mismatched_keys": mismatched},
                             sort_keys=True)
        raise ValueError(f"reference does not cover the complete target state: {details}")

    target.load_state_dict(copied, strict=require_complete)
    architecture = getattr(target, "architecture_contract", None)
    reference_architecture = getattr(reference, "architecture_contract", None)
    return {
        "schema": "tvcg.ablation-initialization.v1",
        "kind": "fresh_cpu_random_reference_by_name_shape_dtype",
        "reference_config_name": reference_config_name,
        "reference_seed": seed,
        "reference_state_sha256": state_fingerprint(source),
        "reference_architecture": reference_architecture() if callable(reference_architecture) else None,
        "copied_state_sha256": state_fingerprint(copied),
        "target_initial_state_sha256": state_fingerprint(target.state_dict()),
        "copied_keys": sorted(copied),
        "missing_keys": sorted(missing),
        "mismatched_keys": mismatched,
        "reference_only_keys": sorted(set(source) - set(destination)),
        "complete_target_coverage": not missing and not mismatched,
        "target_architecture_manifest": architecture() if callable(architecture) else None,
        "checkpoint_loaded": False,
        "optimizer_copied": False,
    }


__all__ = ["initialize_from_reference", "state_fingerprint"]
