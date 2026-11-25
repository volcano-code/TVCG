"""Synthetic contract tests: no actual training/physics claims are made here."""
import copy
import json

import pytest

from experiments.compact_pino_v21.p0_diagnostics.exposure_audit import (
    AuditError,
    ExposureRecorder,
    main,
    read_events,
    summarize_events,
)


def inventory():
    rows = []
    for index, kind in enumerate(("R3", None, "R8", "Full-J", None)):
        rows.append({
            "record_id": f"r{index}", "query_id": f"q{index}",
            "geometry_id": f"g{index % 2}", "source_group_id": f"s{index % 2}",
            "moment_valid": True, "jacobian_valid": kind is not None,
            "loader_eligible": index != 4, "derivative_type": kind,
            "derivative_dimension": {"R3": 3, "R8": 8, "Full-J": 10}.get(kind),
            "chart_id": None if kind is None else f"chart-{kind}",
            "retention_bin": "high" if index % 2 == 0 else "middle",
            "trajectory_stage": "late" if index % 2 == 0 else "general",
            "surface_family": "plane" if kind == "R3" else "curved",
        })
    return {"snapshot_id": "frozen-1756", "manifest_sha256": "a" * 64,
            "registered_count": 10, "records": rows}


def recorder(tmp_path, name="a", **kwargs):
    return ExposureRecorder(tmp_path / f"{name}.jsonl", inventory=inventory(),
                            run_id="pilot", segment_id=name, **kwargs)


def microbatch(audit, batch, window, ids, *, backward=True):
    valid = {row["record_id"]: row["jacobian_valid"] for row in inventory()["records"]}
    audit.forward(batch, window, ids)
    audit.loss(batch, moment_mask=[True] * len(ids), moment_weight=[1.0] * len(ids),
               jacobian_mask=[valid[rid] for rid in ids],
               jacobian_weight=[1.0] * len(ids))
    audit.backward(batch, success=backward,
                   reason=None if backward else "backward_exception")


def complete_step(audit, batch, record, window=None):
    window = window or batch
    microbatch(audit, batch, window, [record])
    audit.optimizer_step(window, committed=True)


def test_prefetch_repetition_masks_and_positive_weights_have_distinct_counts(tmp_path):
    audit = recorder(tmp_path)
    # r3 was prefetched, but never consumed. r0 is genuinely consumed twice.
    audit.prefetch("queued", ["r0", "r0", "r1", "r2", "r3"])
    audit.forward("b", "w", ["r0", "r0", "r1", "r2"])
    audit.loss("b", moment_mask=[True] * 4, moment_weight=[1.0, 0.0, 1.0, 0.0],
               jacobian_mask=[True, True, False, True],
               jacobian_weight=[1.0, 0.0, 1.0, 1.0])
    audit.backward("b", success=True)
    audit.optimizer_step("w", committed=True)
    audit.close()
    report = summarize_events(read_events([audit.path]))
    selected = report["selected_lineage"]
    assert report["status"] == "OBSERVED_COMPLETE"
    assert report["pool"]["registered_candidates"] == 10
    assert report["pool"]["admitted_records"] == 5
    assert report["pool"]["loader_eligible_records"] == 4
    assert selected["counts"]["prefetch_occurrences"] == 5
    assert selected["counts"]["forward_occurrences"] == 4
    assert selected["counts"]["moment_loss_occurrences"] == 2
    assert selected["counts"]["jacobian_loss_occurrences"] == 2
    assert selected["counts"]["committed_moment_occurrences"] == 2
    assert selected["counts"]["committed_jacobian_occurrences"] == 2
    assert selected["unique"]["forward_queries"] == 3
    assert selected["loader_records_not_forwarded"] == ["r3"]
    # Valid zero-valued physical derivatives are not inspected or discarded:
    # qualification and actual loss membership, never numerical nonzero-ness.
    assert selected["jacobian_by_type"]["R3"]["committed_occurrences"] == 1
    assert report["pool"]["per_source_group"]["s0"]["loader_eligible_records"] == 2
    high = [row for row in selected["per_stratum"]
            if row["source_group_id"] == "s0" and row["surface_family"] == "plane"]
    assert high[0]["retention_bin"] == "high"
    assert high[0]["counts"]["forward_occurrences"] == 2
    assert high[0]["counts"]["committed_moment_occurrences"] == 1


def test_gradient_accumulation_counts_three_microbatches_as_one_update(tmp_path):
    audit = recorder(tmp_path)
    microbatch(audit, "b0", "acc", ["r0", "r1"])
    microbatch(audit, "b1", "acc", ["r0", "r2"])
    microbatch(audit, "b2", "acc", ["r1"])
    audit.optimizer_step("acc", committed=True)
    audit.close()
    stats = summarize_events(read_events([audit.path]))["selected_lineage"]
    assert stats["counts"]["optimizer_commits"] == 1
    assert stats["counts"]["forward_occurrences"] == 5
    assert stats["counts"]["committed_moment_occurrences"] == 5
    assert stats["counts"]["committed_jacobian_occurrences"] == 3
    assert stats["unique"]["committed_moment_queries"] == 3


def test_repeated_r8_exposure_does_not_inflate_unique_derivative_records(tmp_path):
    audit = recorder(tmp_path)
    microbatch(audit, "repeat", "first", ["r2", "r2"])
    audit.optimizer_step("first", committed=True)
    complete_step(audit, "again", "r2")
    audit.close()
    stats = summarize_events(read_events([audit.path]))["selected_lineage"]
    assert stats["jacobian_by_type"]["R8"]["committed_occurrences"] == 3
    assert stats["jacobian_by_type"]["R8"]["committed_unique_records"] == 1
    assert stats["jacobian_by_type"]["R3"]["committed_unique_records"] == 0
    assert stats["jacobian_by_type"]["Full-J"]["committed_unique_records"] == 0


def test_amp_skip_and_failed_backward_never_become_committed_exposure(tmp_path):
    audit = recorder(tmp_path)
    microbatch(audit, "amp", "amp", ["r0"])
    audit.optimizer_step("amp", committed=False, reason="amp_nonfinite")
    microbatch(audit, "failed", "failed", ["r0"], backward=False)
    with pytest.raises(AuditError, match="successful backward"):
        audit.optimizer_step("failed", committed=True)
    audit.optimizer_step("failed", committed=False, reason="backward_error")
    # A real optimizer may return None on success. This API takes an explicit
    # success observation, rather than interpreting such a return value.
    complete_step(audit, "ok", "r0")
    audit.close()
    stats = summarize_events(read_events([audit.path]))["selected_lineage"]
    assert stats["counts"]["optimizer_attempts"] == 3
    assert stats["counts"]["optimizer_commits"] == 1
    assert stats["counts"]["optimizer_skips"] == 2
    assert stats["counts"]["moment_loss_occurrences"] == 3
    assert stats["counts"]["moment_backward_occurrences"] == 2
    assert stats["counts"]["committed_moment_occurrences"] == 1
    assert stats["skip_reasons"] == {"amp_nonfinite": 1, "backward_error": 1}


def test_unresolved_accumulation_is_not_an_update_or_a_resumable_checkpoint(tmp_path):
    audit = recorder(tmp_path)
    microbatch(audit, "pending", "acc", ["r0"])
    with pytest.raises(AuditError, match="boundary"):
        audit.checkpoint("unsafe", "b" * 64, optimizer_step=0)
    audit.close(interrupted=True)
    report = summarize_events(read_events([audit.path]))
    assert report["status"] == "INCOMPLETE_EXECUTION_LOG"
    stats = report["selected_lineage"]
    assert stats["counts"]["moment_backward_occurrences"] == 1
    assert stats["counts"]["optimizer_commits"] == 0
    assert stats["counts"]["committed_moment_occurrences"] == 0
    assert stats["unresolved_windows"] == [{"segment_id": "a", "window_id": "acc"}]


def test_resume_rollback_keeps_cost_and_checkpoint_lineage_separate(tmp_path):
    original = recorder(tmp_path, "old", start_step=312,
                        initial_checkpoint_sha256="1" * 64)
    complete_step(original, "kept", "r0")
    parent = original.checkpoint("cp313", "2" * 64, optimizer_step=313)
    complete_step(original, "discarded", "r1")
    original.close()
    resumed = recorder(tmp_path, "resume", resume_from=parent)
    complete_step(resumed, "new", "r2")
    resumed.checkpoint("cp314", "3" * 64, optimizer_step=314)
    resumed.close()
    events = read_events([resumed.path, original.path])
    report = summarize_events(events, selected_checkpoint="cp314")
    assert report["execution"]["counts"]["optimizer_commits"] == 3
    selected = report["selected_lineage"]
    assert selected["initial_step_counter"] == 312
    assert selected["prior_history_inferred"] is False
    assert selected["counts"]["optimizer_commits"] == 2
    assert selected["counts"]["committed_moment_occurrences"] == 2
    assert selected["loader_records_without_committed_supervision"] == ["r1", "r3"]
    earlier = summarize_events(events, selected_checkpoint="cp313")
    assert earlier["selected_lineage"]["counts"]["optimizer_commits"] == 1
    with pytest.raises(AuditError, match="missing parent"):
        summarize_events(read_events([resumed.path]))
    fork = recorder(tmp_path, "fork", resume_from=parent)
    complete_step(fork, "alternative", "r3")
    fork.close()
    branches = events + read_events([fork.path])
    with pytest.raises(AuditError, match="explicitly select"):
        summarize_events(branches)
    chosen = summarize_events(branches, selected_checkpoint="cp314")
    assert chosen["execution"]["counts"]["optimizer_commits"] == 4
    assert chosen["selected_lineage"]["counts"]["optimizer_commits"] == 2


def test_duplicate_events_are_idempotent_but_conflicts_and_gaps_fail(tmp_path):
    audit = recorder(tmp_path)
    complete_step(audit, "ok", "r0")
    audit.close()
    events = read_events([audit.path])
    duplicate = summarize_events(events + copy.deepcopy(events))
    assert duplicate["duplicate_events_ignored"] == len(events)
    assert duplicate["execution"]["counts"]["optimizer_commits"] == 1
    conflict = copy.deepcopy(events[1])
    conflict["record_ids"] = ["r1"]
    with pytest.raises(AuditError, match="conflicting duplicate"):
        summarize_events(events + [conflict])
    with pytest.raises(AuditError, match="sequence"):
        summarize_events(events[:2] + events[3:])


def test_schema_invalid_masks_stale_pool_and_false_commits_are_rejected(tmp_path):
    with pytest.raises(AuditError, match="checkpoint receipt schema"):
        recorder(tmp_path, "bad-resume", resume_from={"kind": "checkpoint"})
    bad_inventory = inventory()
    bad_inventory["records"][0]["derivative_type"] = []
    with pytest.raises(AuditError, match="unsupported derivative_type"):
        ExposureRecorder(tmp_path / "bad-inventory.jsonl", inventory=bad_inventory,
                         run_id="pilot", segment_id="bad")
    audit = recorder(tmp_path)
    with pytest.raises(AuditError, match="loader pool"):
        audit.forward("excluded", "w", ["r4"])
    with pytest.raises(AuditError, match="inventory"):
        audit.forward("missing", "w", ["not-in-snapshot"])
    audit.forward("b", "w", ["r1"])
    with pytest.raises(AuditError, match="invalid jacobian"):
        audit.loss("b", moment_mask=[True], moment_weight=[1.0],
                   jacobian_mask=[True], jacobian_weight=[1.0])
    with pytest.raises(AuditError, match="finite JSON"):
        audit.loss("b", moment_mask=[True], moment_weight=[float("nan")],
                   jacobian_mask=[False], jacobian_weight=[0.0])
    audit.loss("b", moment_mask=[True], moment_weight=[1.0],
               jacobian_mask=[False], jacobian_weight=[0.0])
    with pytest.raises(AuditError, match="successful backward"):
        audit.optimizer_step("w", committed=True)
    audit.backward("b", success=True)
    audit.optimizer_step("w", committed=True)
    with pytest.raises(AuditError, match="checkpoint step"):
        audit.checkpoint("wrong", "b" * 64, optimizer_step=312)
    audit.close()
    assert summarize_events(read_events([audit.path]))["execution"]["counts"]["optimizer_commits"] == 1
    malformed = read_events([audit.path])
    for event in malformed:
        if event["kind"] == "optimizer_step":
            event["optimizer_step"] = True
    with pytest.raises(AuditError, match="optimizer_step must be an integer"):
        summarize_events(malformed)


def test_missing_and_truncated_logs_cli_do_not_infer_1164_or_312(tmp_path, capsys):
    assert summarize_events([])["selected_lineage"] is None
    assert main([]) == 3
    report = json.loads(capsys.readouterr().out)
    assert report["status"] == "INSUFFICIENT_EVIDENCE"
    assert report["execution"] is None
    path = tmp_path / "historical-summary.jsonl"
    path.write_text('{"pool":2158,"exposures":1164,"updates":312}\n', encoding="utf-8")
    output = tmp_path / "audit.json"
    assert main(["--events", str(path), "--output", str(output)]) == 2
    assert json.loads(output.read_text())["selected_lineage"] is None
    path.write_text('{"schema":', encoding="utf-8")
    assert main(["--events", str(path), "--output", str(output)]) == 2
    assert json.loads(output.read_text())["status"] == "INVALID_EVIDENCE"
    original = path.read_text()
    with pytest.raises(SystemExit):
        main(["--events", str(path), "--output", str(path)])
    assert path.read_text() == original

