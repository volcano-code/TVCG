"""Strict, framework-independent training exposure journal.

This module is NOT integrated into the local diagnostic_training/train.py, which
is absent from this repository. A trainer must call this recorder at the actual
event boundaries. The journal audits the caller's assertions, not numerical
gradient correctness, teacher validity, or model promotion.

Minimal integration (all masks/weights are final per-example loss inputs):
    audit.forward("b0", "acc0", ["row0"])
    audit.loss("b0", moment_mask=[True], moment_weight=[1.0],
               jacobian_mask=[False], jacobian_weight=[0.0])
    # loss.backward() has now returned successfully:
    audit.backward("b0", success=True)
    # optimizer.step() has actually executed:
    audit.optimizer_step("acc0", committed=True)

For AMP, determine whether the optimizer actually executed; a None return from
GradScaler.step is not evidence of a skipped update. Log committed=False with a
reason on a real skip. All microbatches in an accumulation window share its ID.
Only boundary checkpoints (no pending accumulated gradients) are supported.
Call checkpoint AFTER successfully saving the corresponding training state.
Resume into a NEW journal file/segment with its returned receipt. Do not append
a restarted process to the old segment or reconstruct missing historic events.

CLI:
    python -m experiments.compact_pino_v21.p0_diagnostics.exposure_audit \
        --events original.jsonl resumed.jsonl --checkpoint checkpoint-id \
        --output exposure_summary.json
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence
from uuid import uuid4

SCHEMA = "compact_pino.exposure.v1"
DERIVATIVE_TYPES = {None, "R3", "R8", "Full-J"}
DIMENSIONS = {"R3": 3, "R8": 8, "Full-J": 10}


class AuditError(ValueError):
    """An incomplete or contradictory assertion must not become evidence."""


def _canonical(value: Any) -> str:
    try:
        return json.dumps(value, ensure_ascii=False, sort_keys=True,
                          separators=(",", ":"), allow_nan=False)
    except (ValueError, TypeError) as exc:
        raise AuditError(f"Not finite JSON data: {exc}") from exc


def _string(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise AuditError(f"{name} must be a nonempty string")
    return value


def _integer(value: Any, name: str, minimum: int = 0) -> int:
    if type(value) is not int or value < minimum:
        raise AuditError(f"{name} must be an integer >= {minimum}")
    return value


def _boolean(value: Any, name: str) -> bool:
    if type(value) is not bool:
        raise AuditError(f"{name} must be boolean")
    return value


def _sha(value: Any, name: str) -> str:
    value = _string(value, name)
    if len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
        raise AuditError(f"{name} must be a lowercase SHA-256")
    return value


def _weight(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise AuditError(f"{name} must be a finite nonnegative number")
    try:
        number = float(value)
    except (OverflowError, ValueError) as exc:
        raise AuditError(f"{name} must be a finite nonnegative number") from exc
    if not math.isfinite(number) or number < 0:
        raise AuditError(f"{name} must be a finite nonnegative number")
    return number


def _checkpoint_receipt(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict) or value.get("kind") != "checkpoint":
        raise AuditError("resume_from must be a recorded checkpoint receipt")
    if value.get("schema") != SCHEMA:
        raise AuditError("checkpoint receipt schema mismatch")
    segment = _string(value.get("segment_id"), "checkpoint segment_id")
    seq = _integer(value.get("seq"), "checkpoint seq", 2)
    if value.get("event_id") != f"{segment}:{seq}":
        raise AuditError("checkpoint event identity mismatch")
    _string(value.get("run_id"), "checkpoint run_id")
    _string(value.get("checkpoint_id"), "checkpoint_id")
    _sha(value.get("checkpoint_sha256"), "checkpoint_sha256")
    _sha(value.get("inventory_sha256"), "inventory_sha256")
    _integer(value.get("optimizer_step"), "checkpoint optimizer_step")
    return value


def _inventory(value: Any) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    if not isinstance(value, dict):
        raise AuditError("inventory must be an object")
    value = copy.deepcopy(value)
    _string(value.get("snapshot_id"), "snapshot_id")
    _sha(value.get("manifest_sha256"), "manifest_sha256")
    registered = _integer(value.get("registered_count"), "registered_count")
    rows = value.get("records")
    if not isinstance(rows, list):
        raise AuditError("inventory.records must be a list of admitted records")
    if registered < len(rows):
        raise AuditError("registered_count is smaller than admitted records")
    records = {}
    for row in rows:
        if not isinstance(row, dict):
            raise AuditError("each inventory record must be an object")
        rid = _string(row.get("record_id"), "record_id")
        if rid in records:
            raise AuditError(f"duplicate inventory record_id: {rid}")
        for key in ("query_id", "geometry_id", "source_group_id"):
            _string(row.get(key), key)
        for key in ("moment_valid", "jacobian_valid", "loader_eligible"):
            _boolean(row.get(key), key)
        if 'residual_valid' in row:
            _boolean(row['residual_valid'], 'residual_valid')
        dtype = row.get("derivative_type")
        if dtype is not None and (not isinstance(dtype, str) or dtype not in DERIVATIVE_TYPES):
            raise AuditError(f"unsupported derivative_type: {dtype!r}")
        if row["jacobian_valid"]:
            if dtype is None:
                raise AuditError("valid Jacobian needs derivative_type")
            _string(row.get("chart_id"), "chart_id")
            dimension = _integer(row.get("derivative_dimension"), "derivative_dimension", 1)
            if dimension != DIMENSIONS[dtype]:
                raise AuditError("derivative_dimension does not match type")
        for key in ("surface_family", "retention_bin", "trajectory_stage"):
            if row.get(key) is not None:
                _string(row[key], key)
        records[rid] = row
    _canonical(value)
    return value, records


def inventory_digest(inventory: Mapping[str, Any]) -> str:
    normalized, _ = _inventory(dict(inventory))
    return hashlib.sha256(_canonical(normalized).encode("utf-8")).hexdigest()


class _State:
    def __init__(self, start: dict[str, Any]):
        if start.get("kind") != "segment_start" or start.get("seq") != 1:
            raise AuditError("each segment must start at seq=1 with segment_start")
        self.inventory, self.records = _inventory(start.get("inventory"))
        self.digest = inventory_digest(self.inventory)
        if start.get("inventory_sha256") != self.digest:
            raise AuditError("inventory digest mismatch")
        self.run_id = _string(start.get("run_id"), "run_id")
        self.segment_id = _string(start.get("segment_id"), "segment_id")
        self.parent = start.get("resume_from")
        self.step = _integer(start.get("start_step"), "start_step")
        if self.parent is not None:
            _checkpoint_receipt(self.parent)
            if self.parent.get("run_id") != self.run_id:
                raise AuditError("resume_from run_id mismatch")
            if self.parent.get("inventory_sha256") != self.digest:
                raise AuditError("resume must use the same frozen inventory")
            if self.parent.get("optimizer_step") != self.step:
                raise AuditError("resume step counter mismatch")
        initial_sha = start.get("initial_checkpoint_sha256")
        if initial_sha is not None:
            _sha(initial_sha, "initial_checkpoint_sha256")
        if self.parent is None and self.step > 0 and initial_sha is None:
            raise AuditError("nonzero starting counter needs initial checkpoint identity")
        self.start = start
        self.seq = 1
        self.closed = False
        self.end_outcome = None
        self.batches: dict[str, dict[str, Any]] = {}
        self.windows: dict[str, dict[str, Any]] = {}
        self.prefetches: list[dict[str, Any]] = []
        self.checkpoints: dict[str, dict[str, Any]] = {}

    @property
    def pending(self) -> list[str]:
        return [wid for wid, window in self.windows.items()
                if window["outcome"] is None]

    def _record_ids(self, value: Any) -> list[str]:
        if not isinstance(value, list) or not value:
            raise AuditError("record_ids must be a nonempty list")
        for rid in value:
            if not isinstance(rid, str) or rid not in self.records:
                raise AuditError(f"record not present in frozen inventory: {rid!r}")
            if not self.records[rid]["loader_eligible"]:
                raise AuditError(f"record is outside actual loader pool: {rid}")
        return value

    def accept(self, event: dict[str, Any]) -> None:
        if self.closed:
            raise AuditError("event after segment_end")
        if event.get("seq") != self.seq + 1:
            raise AuditError("event sequence has a gap or is out of order")
        kind = event.get("kind")
        if kind == "prefetch":
            self._record_ids(event.get("record_ids"))
            _string(event.get("batch_id"), "batch_id")
            self.prefetches.append(event)
        elif kind == "forward":
            bid = _string(event.get("batch_id"), "batch_id")
            wid = _string(event.get("window_id"), "window_id")
            if bid in self.batches:
                raise AuditError("batch_id must be unique within a segment")
            ids = self._record_ids(event.get("record_ids"))
            success = _boolean(event.get("success"), "forward.success")
            if not success:
                _string(event.get("reason"), "forward failure reason")
            window = self.windows.get(wid)
            if window is not None and window["outcome"] is not None:
                raise AuditError("cannot reuse a completed accumulation window")
            self.batches[bid] = {"forward": event, "loss": None, "backward": None}
            if window is None:
                window = {"batch_ids": [], "outcome": None}
                self.windows[wid] = window
            window["batch_ids"].append(bid)
        elif kind == "loss":
            bid = _string(event.get("batch_id"), "batch_id")
            batch = self.batches.get(bid)
            if batch is None or not batch["forward"]["success"]:
                raise AuditError("loss requires a successful recorded forward")
            if batch["loss"] is not None:
                raise AuditError("loss can be recorded only once per batch")
            if self.windows[batch["forward"]["window_id"]]["outcome"] is not None:
                raise AuditError("loss after optimizer outcome")
            rows = event.get("rows")
            ids = batch["forward"]["record_ids"]
            if not isinstance(rows, list) or len(rows) != len(ids):
                raise AuditError("loss rows must match forward occurrences exactly")
            for rid, row in zip(ids, rows):
                if not isinstance(row, dict) or row.get("record_id") != rid:
                    raise AuditError("loss row identity/order does not match forward")
                meta = self.records[rid]
                for prefix, valid in (("moment", "moment_valid"),
                                      ("jacobian", "jacobian_valid")):
                    mask = _boolean(row.get(prefix + "_mask"), prefix + "_mask")
                    _weight(row.get(prefix + "_weight"), prefix + "_weight")
                    if mask and not meta[valid]:
                        raise AuditError(f"invalid {prefix} label has true loss mask")
                for prefix in ('residual', 'constraint'):
                    if prefix + '_mask' in row or prefix + '_weight' in row:
                        mask = _boolean(row.get(prefix + '_mask'), prefix + '_mask')
                        _weight(row.get(prefix + '_weight'), prefix + '_weight')
                        if mask and not meta.get('residual_valid', False):
                            raise AuditError(f'invalid {prefix} target has true loss mask')
            batch["loss"] = event
        elif kind == "backward":
            bid = _string(event.get("batch_id"), "batch_id")
            batch = self.batches.get(bid)
            if batch is None or batch["loss"] is None:
                raise AuditError("backward requires recorded loss membership")
            if batch["backward"] is not None:
                raise AuditError("backward can be recorded only once per batch")
            if self.windows[batch["forward"]["window_id"]]["outcome"] is not None:
                raise AuditError("backward after optimizer outcome")
            if not _boolean(event.get("success"), "backward.success"):
                _string(event.get("reason"), "backward failure reason")
            batch["backward"] = event
        elif kind == "optimizer_step":
            wid = _string(event.get("window_id"), "window_id")
            window = self.windows.get(wid)
            if window is None or window["outcome"] is not None:
                raise AuditError("optimizer outcome needs an open accumulation window")
            committed = _boolean(event.get("committed"), "committed")
            expected = self.step + int(committed)
            if _integer(event.get("optimizer_step"), "optimizer_step") != expected:
                raise AuditError("optimizer step counter is inconsistent with outcome")
            if committed:
                for bid in window["batch_ids"]:
                    backward = self.batches[bid]["backward"]
                    if backward is None or not backward["success"]:
                        raise AuditError("commit requires successful backward for every microbatch")
            else:
                _string(event.get("reason"), "optimizer skip reason")
            window["outcome"] = event
            self.step = expected
        elif kind == "checkpoint":
            _checkpoint_receipt(event)
            if self.pending:
                raise AuditError("checkpoint requires an accumulation boundary")
            cid = _string(event.get("checkpoint_id"), "checkpoint_id")
            _sha(event.get("checkpoint_sha256"), "checkpoint_sha256")
            if cid in self.checkpoints:
                raise AuditError("checkpoint_id must be immutable and unique")
            if event.get("optimizer_step") != self.step:
                raise AuditError("checkpoint step differs from recorded optimizer state")
            if event.get("inventory_sha256") != self.digest:
                raise AuditError("checkpoint inventory mismatch")
            self.checkpoints[cid] = event
        elif kind == "segment_end":
            outcome = event.get("outcome")
            if outcome not in {"completed", "interrupted"}:
                raise AuditError("segment end must be completed or interrupted")
            if outcome == "completed" and self.pending:
                raise AuditError("completed segment cannot contain unresolved accumulation")
            self.closed = True
            self.end_outcome = outcome
        else:
            raise AuditError(f"unknown event kind: {kind!r}")
        self.seq = event["seq"]


class ExposureRecorder:
    """Single-process JSONL recorder; create one new file per process segment.

    inventory contains snapshot_id, manifest_sha256, registered_count, and
    records. Each admitted record has record_id, query_id, geometry_id,
    source_group_id, moment_valid, jacobian_valid, loader_eligible,
    derivative_type, chart_id, and derivative_dimension (required for valid J).
    Optional strata such as retention_bin and trajectory_stage are preserved.

    A nonzero start_step records only a baseline counter, NEVER its historical
    exposures. Resume receipts require the parent journal during CLI analysis.
    Record only successful saving via checkpoint; this module never saves,
    updates, loads, or validates the trainer's weights.
    """

    def __init__(self, path: str | Path, *, inventory: dict[str, Any],
                 run_id: str, segment_id: str | None = None,
                 resume_from: dict[str, Any] | None = None,
                 start_step: int | None = None,
                 initial_checkpoint_sha256: str | None = None,
                 durable: bool = False):
        inventory, _ = _inventory(inventory)
        self.path = Path(path)
        self.run_id = _string(run_id, "run_id")
        self.segment_id = _string(segment_id or uuid4().hex, "segment_id")
        if resume_from is not None:
            _checkpoint_receipt(resume_from)
        if start_step is None:
            start_step = resume_from["optimizer_step"] if resume_from else 0
        start = self._envelope(1, "segment_start", {
            "inventory": inventory, "inventory_sha256": inventory_digest(inventory),
            "resume_from": copy.deepcopy(resume_from), "start_step": start_step,
            "initial_checkpoint_sha256": initial_checkpoint_sha256})
        self._state = _State(start)
        self._durable = durable
        self._broken = False
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._stream = self.path.open("x", encoding="utf-8")
        self._write(start)

    def _envelope(self, seq: int, kind: str, payload: dict[str, Any]) -> dict[str, Any]:
        return {"schema": SCHEMA, "run_id": self.run_id,
                "segment_id": self.segment_id, "seq": seq,
                "event_id": f"{self.segment_id}:{seq}",
                "time_utc": datetime.now(timezone.utc).isoformat(),
                "kind": kind, **copy.deepcopy(payload)}

    def _write(self, event: dict[str, Any]) -> None:
        try:
            self._stream.write(_canonical(event) + "\n")
            self._stream.flush()
            if self._durable or event["kind"] in {"checkpoint", "segment_end"}:
                os.fsync(self._stream.fileno())
        except Exception:
            self._broken = True
            raise

    def _emit(self, kind: str, payload: dict[str, Any]) -> dict[str, Any]:
        if self._broken or self._stream.closed:
            raise AuditError("journal is unavailable; do not continue unaudited")
        event = self._envelope(self._state.seq + 1, kind, payload)
        _canonical(event)
        self._state.accept(event)
        self._write(event)
        return copy.deepcopy(event)

    def prefetch(self, batch_id: str, record_ids: Sequence[str]) -> None:
        """Optional debugging event. It never counts as training exposure."""
        self._emit("prefetch", {"batch_id": batch_id, "record_ids": list(record_ids)})

    def forward(self, batch_id: str, window_id: str, record_ids: Sequence[str],
                *, success: bool = True, reason: str | None = None) -> None:
        self._emit("forward", {"batch_id": batch_id, "window_id": window_id,
                              "record_ids": list(record_ids),
                              "success": success, "reason": reason})

    def loss(self, batch_id: str, *, moment_mask: Sequence[bool],
             moment_weight: Sequence[float], jacobian_mask: Sequence[bool],
             jacobian_weight: Sequence[float], residual_mask=None, residual_weight=None,
             constraint_mask=None, constraint_weight=None) -> None:
        if batch_id not in self._state.batches:
            raise AuditError("loss requires a recorded batch")
        ids = self._state.batches[batch_id]["forward"]["record_ids"]
        values = [list(moment_mask), list(moment_weight),
                  list(jacobian_mask), list(jacobian_weight)]
        if any(len(value) != len(ids) for value in values):
            raise AuditError("mask/weight lengths must equal batch occurrences")
        rows = [{"record_id": rid, "moment_mask": mm, "moment_weight": mw,
                 "jacobian_mask": jm, "jacobian_weight": jw}
                for rid, mm, mw, jm, jw in zip(ids, *values)]
        for prefix, masks, weights in (('residual', residual_mask, residual_weight),
                                        ('constraint', constraint_mask, constraint_weight)):
            if masks is None and weights is None:
                continue
            if masks is None or weights is None or len(masks) != len(ids) or len(weights) != len(ids):
                raise AuditError('physical mask/weight lengths must equal batch occurrences')
            for row, mask, weight in zip(rows, masks, weights):
                row[prefix + '_mask'] = mask
                row[prefix + '_weight'] = weight
        self._emit("loss", {"batch_id": batch_id, "rows": rows})

    def backward(self, batch_id: str, *, success: bool,
                 reason: str | None = None) -> None:
        self._emit("backward", {"batch_id": batch_id,
                               "success": success, "reason": reason})

    def optimizer_step(self, window_id: str, *, committed: bool,
                       reason: str | None = None) -> None:
        _boolean(committed, "committed")
        self._emit("optimizer_step", {
            "window_id": window_id, "committed": committed, "reason": reason,
            "optimizer_step": self._state.step + int(committed)})

    def checkpoint(self, checkpoint_id: str, checkpoint_sha256: str, *,
                   optimizer_step: int) -> dict[str, Any]:
        """Assert an already-saved boundary checkpoint; return resume receipt."""
        return self._emit("checkpoint", {
            "checkpoint_id": checkpoint_id, "checkpoint_sha256": checkpoint_sha256,
            "optimizer_step": optimizer_step,
            "inventory_sha256": self._state.digest})

    def close(self, *, interrupted: bool = False) -> None:
        if self._stream.closed:
            return
        try:
            self._emit("segment_end", {
                "outcome": "interrupted" if interrupted else "completed"})
        finally:
            self._stream.close()


def read_events(paths: Sequence[str | Path]) -> list[dict[str, Any]]:
    result = []
    for path in paths:
        with Path(path).open(encoding="utf-8") as stream:
            for line_number, line in enumerate(stream, 1):
                if not line.strip():
                    continue
                try:
                    value = json.loads(line)
                    _canonical(value)
                except (ValueError, AuditError) as exc:
                    raise AuditError(f"{path}:{line_number}: invalid JSON: {exc}") from exc
                if not isinstance(value, dict):
                    raise AuditError(f"{path}:{line_number}: event is not an object")
                result.append(value)
    return result


def _parse(events: Iterable[dict[str, Any]]) -> tuple[dict[str, _State], int]:
    seen = {}
    grouped: dict[str, list[dict[str, Any]]] = {}
    duplicates = 0
    for event in events:
        if not isinstance(event, dict) or event.get("schema") != SCHEMA:
            raise AuditError("unsupported or missing event schema")
        segment = _string(event.get("segment_id"), "segment_id")
        seq = _integer(event.get("seq"), "seq", 1)
        _string(event.get("run_id"), "run_id")
        eid = f"{segment}:{seq}"
        if event.get("event_id") != eid:
            raise AuditError("event_id does not match segment and sequence")
        normalized = _canonical(event)
        if eid in seen:
            if seen[eid] != normalized:
                raise AuditError(f"conflicting duplicate event: {eid}")
            duplicates += 1
            continue
        seen[eid] = normalized
        grouped.setdefault(segment, []).append(copy.deepcopy(event))
    states = {}
    for segment, values in grouped.items():
        values.sort(key=lambda item: item["seq"])
        state = _State(values[0])
        for event in values[1:]:
            if event["run_id"] != state.run_id:
                raise AuditError("run_id changed inside a segment")
            state.accept(event)
        states[segment] = state
    if len({state.run_id for state in states.values()}) > 1:
        raise AuditError("analyze independent run_ids separately")
    if len({state.digest for state in states.values()}) > 1:
        raise AuditError("frozen inventory changed across run segments")
    checkpoints = {}
    for state in states.values():
        for cid, receipt in state.checkpoints.items():
            if cid in checkpoints:
                raise AuditError("checkpoint_id reused across segments")
            checkpoints[cid] = receipt
    for state in states.values():
        parent = state.parent
        if parent is None:
            continue
        actual = checkpoints.get(parent.get("checkpoint_id"))
        if actual is None:
            raise AuditError("missing parent checkpoint journal; cannot infer historic updates")
        if _canonical(actual) != _canonical(parent):
            raise AuditError("resume receipt differs from the recorded checkpoint")
        if parent["segment_id"] == state.segment_id:
            raise AuditError("segment cannot resume from itself")
    # Follow ancestry to reject cycles even when every receipt is present.
    for segment in states:
        visited = set()
        current = segment
        while current is not None:
            if current in visited:
                raise AuditError("cycle in resume lineage")
            visited.add(current)
            parent = states[current].parent
            current = parent["segment_id"] if parent is not None else None
    return states, duplicates


def _pool_counts(state: _State) -> dict[str, Any]:
    rows = list(state.records.values())
    loader = [row for row in rows if row["loader_eligible"]]
    groups = {}
    for source in sorted({row["source_group_id"] for row in rows}):
        admitted = [row for row in rows if row["source_group_id"] == source]
        eligible = [row for row in admitted if row["loader_eligible"]]
        groups[source] = {
            "admitted_records": len(admitted), "loader_eligible_records": len(eligible),
            "moment_eligible_records": sum(row["moment_valid"] for row in eligible),
            "jacobian_eligible_records": sum(row["jacobian_valid"] for row in eligible)}
    return {
        "registered_candidates": state.inventory["registered_count"],
        "admitted_records": len(rows), "loader_eligible_records": len(loader),
        "moment_eligible_records": sum(row["moment_valid"] for row in loader),
        "jacobian_eligible_records": sum(row["jacobian_valid"] for row in loader),
        "jacobian_eligible_by_type": {
            kind: sum(row["jacobian_valid"] and row["derivative_type"] == kind
                      for row in loader) for kind in DIMENSIONS},
        "per_source_group": groups,
    }


def _statistics(states: dict[str, _State],
                selected: list[tuple[str, int]]) -> dict[str, Any]:
    counts: Counter[str] = Counter()
    unique: dict[str, set[Any]] = {
        key: set() for key in ("forward_queries", "moment_loss_queries",
                              "jacobian_loss_records", "committed_moment_queries",
                              "committed_jacobian_records", "forward_records",
                              "committed_supervision_records", 'residual_loss_queries',
                              'constraint_loss_queries', 'committed_residual_queries',
                              'committed_constraint_queries')}
    inventory_state = next(iter(states.values()))
    groups: dict[str, Counter[str]] = {
        row["source_group_id"]: Counter() for row in inventory_state.records.values()}
    strata: dict[tuple[str, ...], Counter[str]] = {}
    derivative_counts = {kind: Counter() for kind in DIMENSIONS}
    committed_derivative_records = {kind: set() for kind in DIMENSIONS}
    skipped_reasons: Counter[str] = Counter()
    pending = []
    for segment, cut in selected:
        state = states[segment]
        for event in state.prefetches:
            if event["seq"] <= cut:
                counts["prefetch_occurrences"] += len(event["record_ids"])
        for wid, window in state.windows.items():
            bids = [bid for bid in window["batch_ids"]
                    if state.batches[bid]["forward"]["seq"] <= cut]
            if not bids:
                continue
            outcome = window["outcome"]
            if outcome is not None and outcome["seq"] > cut:
                outcome = None
            committed = outcome is not None and outcome["committed"]
            if outcome is None:
                pending.append({"segment_id": segment, "window_id": wid})
            else:
                counts["optimizer_attempts"] += 1
                counts["optimizer_commits" if committed else "optimizer_skips"] += 1
                if not committed:
                    skipped_reasons[outcome["reason"]] += 1
            window_supervision = 0
            for bid in bids:
                batch = state.batches[bid]
                forward = batch["forward"]
                counts["forward_attempt_occurrences"] += len(forward["record_ids"])
                if not forward["success"]:
                    counts["forward_failed_occurrences"] += len(forward["record_ids"])
                    continue
                loss = batch["loss"]
                backward = batch["backward"]
                if loss is not None and loss["seq"] > cut:
                    loss = None
                if backward is not None and backward["seq"] > cut:
                    backward = None
                for index, rid in enumerate(forward["record_ids"]):
                    row = state.records[rid]
                    query = (row["geometry_id"], row["query_id"])
                    derivative = (query, row.get("chart_id"),
                                  row.get("derivative_type"))
                    group = groups.setdefault(row["source_group_id"], Counter())
                    stratum_key = (row["source_group_id"],) + tuple(
                        row.get(key) or "UNSPECIFIED"
                        for key in ("surface_family", "retention_bin", "trajectory_stage"))
                    stratum = strata.setdefault(stratum_key, Counter())
                    counts["forward_occurrences"] += 1
                    group["forward_occurrences"] += 1
                    stratum["forward_occurrences"] += 1
                    unique["forward_queries"].add(query)
                    unique["forward_records"].add(rid)
                    if loss is None:
                        continue
                    membership = loss["rows"][index]
                    for name, unique_name in (
                        ("moment", "moment_loss_queries"),
                        ("jacobian", "jacobian_loss_records"),
                        ('residual', 'residual_loss_queries'),
                        ('constraint', 'constraint_loss_queries')):
                        included = (membership.get(name + "_mask", False) and
                                    membership.get(name + "_weight", 0.) > 0)
                        if not included:
                            continue
                        counts[name + "_loss_occurrences"] += 1
                        group[name + "_loss_occurrences"] += 1
                        stratum[name + "_loss_occurrences"] += 1
                        unique[unique_name].add(derivative if name == "jacobian" else query)
                        dtype = row["derivative_type"]
                        if name == "jacobian":
                            derivative_counts[dtype]["loss_occurrences"] += 1
                        if backward is not None and backward["success"]:
                            counts[name + "_backward_occurrences"] += 1
                            if name == "jacobian":
                                derivative_counts[dtype]["backward_occurrences"] += 1
                        if committed:
                            window_supervision += 1
                            counts["committed_" + name + "_occurrences"] += 1
                            group["committed_" + name + "_occurrences"] += 1
                            stratum["committed_" + name + "_occurrences"] += 1
                            unique["committed_supervision_records"].add(rid)
                            unique["committed_" + name +
                                   ("_records" if name == "jacobian" else "_queries")].add(
                                       derivative if name == "jacobian" else query)
                            if name == "jacobian":
                                derivative_counts[dtype]["committed_occurrences"] += 1
                                committed_derivative_records[dtype].add(derivative)
            if committed and window_supervision == 0:
                counts["commits_without_supervision"] += 1
    required = (
        "prefetch_occurrences", "forward_attempt_occurrences",
        "forward_failed_occurrences", "forward_occurrences",
        "moment_loss_occurrences", "jacobian_loss_occurrences",
        "moment_backward_occurrences", "jacobian_backward_occurrences",
        "committed_moment_occurrences", "committed_jacobian_occurrences",
        'residual_loss_occurrences', 'constraint_loss_occurrences',
        'residual_backward_occurrences', 'constraint_backward_occurrences',
        'committed_residual_occurrences', 'committed_constraint_occurrences',
        "optimizer_attempts", "optimizer_commits", "optimizer_skips",
        "commits_without_supervision")
    for key in required:
        counts.setdefault(key, 0)
    group_keys = ("forward_occurrences", "moment_loss_occurrences", "jacobian_loss_occurrences",
                  "committed_moment_occurrences", "committed_jacobian_occurrences",
                  'residual_loss_occurrences', 'constraint_loss_occurrences',
                  'committed_residual_occurrences', 'committed_constraint_occurrences')
    for group in list(groups.values()) + list(strata.values()):
        for key in group_keys:
            group.setdefault(key, 0)
    for group in derivative_counts.values():
        for key in ("loss_occurrences", "backward_occurrences", "committed_occurrences"):
            group.setdefault(key, 0)
    state = next(iter(states.values()))
    eligible = {rid for rid, row in state.records.items() if row["loader_eligible"]}
    return {
        "counts": dict(counts),
        "unique": {key: len(value) for key, value in unique.items()},
        "per_source_group": {key: dict(value) for key, value in sorted(groups.items())},
        "per_stratum": [dict(zip(("source_group_id", "surface_family", "retention_bin",
                                  "trajectory_stage"), key), counts=dict(value))
                        for key, value in sorted(strata.items())],
        "jacobian_by_type": {
            key: {**dict(value), "committed_unique_records":
                  len(committed_derivative_records[key])}
            for key, value in derivative_counts.items()},
        "skip_reasons": dict(skipped_reasons),
        "unresolved_windows": pending,
        "loader_records_not_forwarded": sorted(eligible - unique["forward_records"]),
        "loader_records_without_committed_supervision": sorted(
            eligible - unique["committed_supervision_records"]),
    }


def summarize_events(events: Iterable[dict[str, Any]], *,
                     selected_segment: str | None = None,
                     selected_checkpoint: str | None = None) -> dict[str, Any]:
    """Count observed events, with resume rollback excluded from model lineage.

    Missing parent journals and contradictory/missing sequence events raise
    AuditError. An entirely absent log returns INSUFFICIENT_EVIDENCE with null
    counts rather than inventing zero or deriving updates from exposure totals.
    """
    if selected_segment is not None and selected_checkpoint is not None:
        raise AuditError("select either segment or checkpoint")
    states, duplicates = _parse(events)
    if not states:
        return {"schema": SCHEMA, "status": "INSUFFICIENT_EVIDENCE",
                "execution": None, "selected_lineage": None,
                "promotion_status": "NOT_EVALUATED",
                "reason": "No event evidence; historical updates cannot be reconstructed."}
    cut = None
    if selected_checkpoint is not None:
        found = [(sid, receipt) for sid, state in states.items()
                 for cid, receipt in state.checkpoints.items() if cid == selected_checkpoint]
        if not found:
            raise AuditError("selected checkpoint is absent from journals")
        selected_segment, receipt = found[0]
        cut = receipt["seq"]
    elif selected_segment is None:
        parents = {state.parent["segment_id"] for state in states.values()
                   if state.parent is not None}
        leaves = set(states) - parents
        if len(leaves) != 1:
            raise AuditError("multiple terminal segments: explicitly select model lineage")
        selected_segment = leaves.pop()
    if selected_segment not in states:
        raise AuditError("selected segment is absent from journals")
    if cut is None:
        cut = states[selected_segment].seq
    selection = []
    segment = selected_segment
    while segment is not None:
        selection.append((segment, cut))
        parent = states[segment].parent
        if parent is None:
            break
        segment, cut = parent["segment_id"], parent["seq"]
    selection.reverse()
    root = states[selection[0][0]]
    incomplete = [sid for sid, state in states.items()
                  if state.end_outcome != "completed"]
    return {
        "schema": SCHEMA,
        "status": "OBSERVED_COMPLETE" if not incomplete else "INCOMPLETE_EXECUTION_LOG",
        "promotion_status": "NOT_EVALUATED",
        "run_id": root.run_id, "snapshot_id": root.inventory["snapshot_id"],
        "manifest_sha256": root.inventory["manifest_sha256"],
        "inventory_sha256": root.digest, "pool": _pool_counts(root),
        "duplicate_events_ignored": duplicates, "incomplete_segments": incomplete,
        "execution": _statistics(states, [(sid, state.seq)
                                           for sid, state in states.items()]),
        "selected_lineage": {
            "selection": {"segment_id": selected_segment,
                          "checkpoint_id": selected_checkpoint},
            "initial_step_counter": root.start["start_step"],
            "initial_checkpoint_sha256": root.start.get("initial_checkpoint_sha256"),
            "prior_history_inferred": False,
            "segments": [{"segment_id": sid, "through_seq": seq} for sid, seq in selection],
            **_statistics(states, selection)},
        "limitations": [
            "Counts cover recorded caller assertions only, not unlogged training.",
            "Positive masked loss participation does not prove nonzero gradients.",
            "Physical labels, checkpoint bytes, and model promotion are not verified.",
            "Execution totals include discarded branches; selected lineage excludes them.",
            "A missing optimizer outcome remains unresolved, never assumed committed.",
        ],
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--events", nargs="*", default=[], type=Path)
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--segment")
    group.add_argument("--checkpoint")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    if args.output is not None and args.output.resolve() in {
            path.resolve() for path in args.events}:
        parser.error("--output must not overwrite an input journal")
    try:
        report = summarize_events(read_events(args.events),
                                  selected_segment=args.segment,
                                  selected_checkpoint=args.checkpoint)
        code = 0 if report["status"] == "OBSERVED_COMPLETE" else 3
    except (AuditError, OSError) as exc:
        report = {"schema": SCHEMA, "status": "INVALID_EVIDENCE",
                  "execution": None, "selected_lineage": None,
                  "promotion_status": "NOT_EVALUATED", "reason": str(exc)}
        code = 2
    text = json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    if args.output is None:
        print(text, end="")
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf-8")
    return code


if __name__ == "__main__":
    raise SystemExit(main())

