"""Fail-closed mapper from an audited Spin-It initial-state export to M69 context.

No geometry semantics are inferred. The record must explicitly contain the
initial full-material moments, normalized quadric anchor, exact TaskSpec
payload, active mask and length scale from one provenance-bound run.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from typing import Any, Mapping

import torch
from torch import Tensor

from .unified_faster_bridge import M69GyroscopeContext


SPIN_INITIAL_STATE_SCHEMA = "tvcg.spin-it.initial-gyroscope-state.v1"
M69_TASK_SCHEMA = "m69.fixed-axis-task.v1"


@dataclass(frozen=True)
class MappedInitialState:
    context: M69GyroscopeContext
    source_geometry_sha256: str
    source_state_sha256: str
    source_task_hash: str
    source_evidence: str


def _sha256_json(value: Any) -> str:
    raw = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _required_hash(value: Any, name: str) -> str:
    if not isinstance(value, str) or len(value) != 64:
        raise ValueError(f"{name} must be a 64-character SHA-256 hex string")
    try:
        int(value, 16)
    except ValueError as exc:
        raise ValueError(f"{name} must be hexadecimal") from exc
    return value.lower()


def _finite_tensor(value: Any, shape: tuple[int, ...], name: str, *,
                   dtype: torch.dtype, device: torch.device) -> Tensor:
    tensor = torch.as_tensor(value, dtype=dtype, device=device)
    if tuple(tensor.shape) != shape or not torch.isfinite(tensor).all():
        raise ValueError(f"{name} must be finite with shape {shape}")
    return tensor


def map_spin_initial_state_record(
    record: Mapping[str, Any],
    *,
    device: str | torch.device = "cpu",
    dtype: torch.dtype = torch.float64,
) -> MappedInitialState:
    """Map one fully audited export; fail closed on missing or ambiguous fields.

    Required source record fields:
      schema, status, evidence, source_geometry_sha256, state_payload_sha256,
      task_payload, task_payload_sha256, initial_state.

    initial_state must explicitly provide beta_anchor, original_moments,
    length_scale, active_mask, axis, axis_origin, contact, and coordinate_frame.
    Axis/origin/contact are checked against the exact registered TaskSpec values;
    no axis is inferred from a mesh principal axis or target-frame default.
    """
    if record.get("schema") != SPIN_INITIAL_STATE_SCHEMA:
        raise ValueError("unsupported Spin-It initial-state schema")
    if record.get("status") != "AUDITED_INITIAL_STATE":
        raise ValueError("Spin-It record is not admitted as AUDITED_INITIAL_STATE")
    evidence = record.get("evidence")
    if evidence not in {"REPRODUCED_SOURCE", "AUDITED_REAL_GEOMETRY"}:
        raise ValueError("reconstructed/demo assets cannot be admitted as training states")

    geometry_hash = _required_hash(record.get("source_geometry_sha256"), "source_geometry_sha256")
    state_hash = _required_hash(record.get("state_payload_sha256"), "state_payload_sha256")
    task_hash = _required_hash(record.get("task_payload_sha256"), "task_payload_sha256")
    task = record.get("task_payload")
    state = record.get("initial_state")
    if not isinstance(task, Mapping) or not isinstance(state, Mapping):
        raise ValueError("task_payload and initial_state must be mappings")
    if _sha256_json(task) != task_hash:
        raise ValueError("TaskSpec payload SHA-256 mismatch")
    if _sha256_json(state) != state_hash:
        raise ValueError("initial-state payload SHA-256 mismatch")
    if task.get("schema") != M69_TASK_SCHEMA:
        raise ValueError("unsupported M69 TaskSpec schema")

    required_task = (
        "axis", "axis_origin", "contact", "density", "polar_retention",
        "min_removal_fraction", "max_centroid_rise", "com_tolerance",
        "alignment_tolerance", "principal_gap_min", "require_connected",
        "mass_tolerance", "inertia_tolerance", "moment_psd_tolerance",
        "solver_tolerance", "objective", "material_policy", "inertia_reference",
    )
    missing_task = [key for key in required_task if key not in task]
    if missing_task:
        raise ValueError("TaskSpec missing required reviewed fields: " + ",".join(missing_task))

    required_state = (
        "beta_anchor", "original_moments", "length_scale", "active_mask",
        "axis", "axis_origin", "contact", "coordinate_frame",
        "fixed_material_sha256", "design_domain_sha256",
    )
    missing_state = [key for key in required_state if key not in state]
    if missing_state:
        raise ValueError("initial state missing required fields: " + ",".join(missing_state))

    if state["coordinate_frame"] != "task_frame":
        raise ValueError("initial-state vectors must already be expressed in the registered task_frame")
    for key in ("axis", "axis_origin", "contact"):
        if state[key] != task[key]:
            raise ValueError(f"initial-state {key} does not match registered TaskSpec; explicit transform required")

    device = torch.device(device)
    beta = _finite_tensor(state["beta_anchor"], (10,), "beta_anchor", dtype=dtype, device=device)
    moments = _finite_tensor(state["original_moments"], (10,), "original_moments", dtype=dtype, device=device)
    active = torch.as_tensor(state["active_mask"], dtype=torch.bool, device=device)
    if active.shape != (10,):
        raise ValueError("active_mask must have shape [10]")
    length_scale = float(state["length_scale"])
    if not math.isfinite(length_scale) or length_scale <= 0:
        raise ValueError("length_scale must be finite and positive")
    _required_hash(state["fixed_material_sha256"], "fixed_material_sha256")
    _required_hash(state["design_domain_sha256"], "design_domain_sha256")

    context = M69GyroscopeContext(
        beta_anchor=beta,
        original_moments=moments,
        task=dict(task),
        length_scale=length_scale,
        active_mask=active,
    )
    context.validate()
    return MappedInitialState(
        context=context,
        source_geometry_sha256=geometry_hash,
        source_state_sha256=state_hash,
        source_task_hash=task_hash,
        source_evidence=evidence,
    )


__all__ = [
    "SPIN_INITIAL_STATE_SCHEMA",
    "M69_TASK_SCHEMA",
    "MappedInitialState",
    "map_spin_initial_state_record",
]
