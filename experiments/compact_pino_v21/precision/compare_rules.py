"""Compare full-cell rules and an optional frozen model at identical queries.

Every rule retains evaluate_packed(model=None) results at the original pitch
and h/pitch=1. An explicitly hashed current-runner checkpoint adds paired
learned_rules, sharing encoded geometry and the same metrics. Native labels
enter metrics only after prediction. This tool neither trains a network nor
evaluates a solver optimum or the original 5% acceptance gate. A missing
qualified derivative or bound physical profile remains explicitly unavailable.

Examples (run from the repository root):
  python -m experiments.compact_pino_v21.precision.compare_rules \
      --manifest export/training_manifest.json --mode trial --output rules.json
  python -m experiments.compact_pino_v21.precision.compare_rules \
      --case synthetic/case.json --rules tetra4 gl2 gl3 --output fixture.json
  python -m experiments.compact_pino_v21.precision.compare_rules \
      --manifest export/training_manifest.json --mode trial \
      --checkpoint run/step.pt --checkpoint-sha256 SHA256 --device cuda:0 \
      --output paired_rules.json
"""
from __future__ import annotations

import argparse
from importlib import import_module
import json
from pathlib import Path
import platform
import time

import numpy as np

from ..p0_diagnostics.baseline_cases import (
    PREFIX, finite_array, jsonable, load_baseline_case, operand_fingerprint,
    query_fingerprint, sha256, validate_case,
)
from ..p0_diagnostics.deterministic_quadrature import _audit_blocks
from ..p0_diagnostics.export_cases import verify_export_qualification


RULES = ("tetra4", "gl2", "gl3")
SCHEMA = "tvcg.v22.common-query-cubature.v1"
CASE_SCHEMA = "tvcg.v22.common-query-cubature-case.v1"


def _options(rules, chunk_size, repeats, warmup):
    rules = tuple(rules)
    if not rules or len(set(rules)) != len(rules) or any(rule not in RULES for rule in rules):
        raise ValueError("rules must be a nonempty unique selection of tetra4, gl2, gl3")
    if any(type(n) is not int or n < 1 for n in (chunk_size, repeats)):
        raise ValueError("positive integer chunk_size and repeats are required")
    if type(warmup) is not int or warmup < 0:
        raise ValueError("warmup must be a nonnegative integer")
    return rules


def _moment_metrics(predicted, reference, scale):
    delta = predicted - reference
    normalized = delta / scale
    return dict(raw_delta=delta, raw_l2=float(np.linalg.norm(delta)),
                raw_rmse=float(np.sqrt(np.mean(delta ** 2))),
                raw_max_abs=float(np.max(np.abs(delta))),
                scaled_delta=normalized, scaled_l2=float(np.linalg.norm(normalized)),
                scaled_rmse=float(np.sqrt(np.mean(normalized ** 2))),
                scaled_max_abs=float(np.max(np.abs(normalized))))


def _derivative_metrics(prediction, case, scale):
    meta = case["meta"]
    dimension = meta["derivative_dimension"]
    base = dict(dimension=dimension, chart_id=meta["chart_id"],
                derivative_type={3: "R3", 8: "R8", 10: "Full-J"}.get(dimension),
                qualified_reference=meta["derivative_valid"],
                chart_basis=None, parameter_scales=None, predicted_jacobian=None,
                native_jacobian=None, raw_delta=None, dimensionless_delta=None,
                chart_unit_axes=None, native_direction=None)
    if not meta["derivative_valid"]:
        return dict(base, availability="UNAVAILABLE_NATIVE_REFERENCE",
                    reason=meta.get("derivative_reason", "derivative label is not qualified"))
    if prediction.jacobian is None:
        return dict(base, availability="UNAVAILABLE_RESPONSE_DERIVATIVE",
                    reason="the finite-response active distance is nonregular")
    jacobian = finite_array(prediction.jacobian.detach().cpu().numpy(), (10, dimension), "predicted Jacobian")
    native = finite_array(case["native_jacobian"], (10, dimension), "qualified native Jacobian")
    parameters = finite_array(case["parameter_scales"], (dimension,), "parameter scales")
    delta = jacobian - native
    directions = np.diag(parameters)
    # Column j is v_j = parameter_scale[j] e_j; ||v_j/scales||_2 == 1.
    # These directions are fixed before observing candidate errors.
    jv = jacobian @ directions
    native_jv = native @ directions
    scaled = (jv - native_jv) / scale[:, None]
    base.update(availability="AVAILABLE", reason=None, chart_basis=case["chart_basis"],
                parameter_scales=parameters, predicted_jacobian=jacobian,
                native_jacobian=native, raw_delta=delta, dimensionless_delta=scaled,
                chart_unit_axes=dict(selection="predeclared_all_chart_unit_axes",
                                     normalization="v_j = parameter_scales[j] * e_j; norm(v_j / parameter_scales) = 1",
                                     directions=directions, predicted_Jv=jv, native_Jv=native_jv,
                                     raw_delta=jv-native_jv, scaled_delta=scaled,
                                     scaled_l2_per_direction=np.linalg.norm(scaled, axis=0)))
    if case.get("direction") is None:
        base["native_direction"] = dict(availability="UNAVAILABLE_NATIVE_DIRECTION",
                                         reason="no recorded native Newton direction was supplied")
    else:
        direction = finite_array(case["direction"], (dimension,), "recorded native direction")
        norm = float(np.linalg.norm(direction / parameters))
        base["native_direction"] = dict(
            availability="AVAILABLE" if norm > 0 else "ZERO_DIRECTION_NOT_INFORMATIVE",
            source=meta["direction_source"], direction=direction, parameter_norm=norm,
            predicted_Jv=jacobian @ direction, native_Jv=native @ direction,
            raw_delta=delta @ direction, scaled_delta=delta @ direction / scale,
            unit_direction=None if norm == 0 else direction / norm,
            unit_scaled_delta=None if norm == 0 else delta @ (direction / norm) / scale)
    return base


def _validate_bound_target(case, target):
    if target is None:
        return
    if isinstance(target, dict) and target.get("valid") is False:
        if not isinstance(target.get("reason"), str) or not target["reason"].strip():
            raise ValueError("unavailable residual target requires its reason")
        # The manifest accepts an explicit absence declaration without a
        # numerical-target schema or case binding. No physical metric uses it.
        return
    expected = case.get("manifest_receipt", {}).get("sha256")
    if not expected or not isinstance(target, dict) or target.get("binding", {}).get("case_sha256") != expected:
        raise ValueError("physical residual target must be bound to this exact case SHA256")
    from ..training.residual_matching import validate_residual_target
    validate_residual_target(target, case["native_moments"], coefficients=case["coefficients"])


def _physical_metrics(predicted, case, target):
    if target is None:
        return dict(availability="UNAVAILABLE_PROFILE",
                    reason="no qualified case-bound fixed-reference-multiplier profile was supplied",
                    native_solver_acceptance="NOT_EVALUATED")
    if isinstance(target, dict) and target.get("valid") is False:
        return dict(availability="UNAVAILABLE_TARGET", reason=target["reason"],
                    native_solver_acceptance="NOT_EVALUATED")
    from ..training.residual_matching import residual_metrics
    result = residual_metrics(predicted, case["native_moments"], target, coefficients=case["coefficients"])
    result["native_solver_acceptance"] = "NOT_EVALUATED"
    return result


def _measure_rule(evaluate, case, scale, rule, *, repeats, warmup, residual_target,
                  synchronize=None, model_weights_used=False, device="cpu"):
    """Shared numerical/physical metrics for no-learning and learned rows."""
    sync = synchronize if synchronize is not None else lambda: None
    blocks, operand = case["blocks"], case["operand"]
    material = getattr(blocks, "material", None)
    sync()
    started = time.perf_counter()
    for _ in range(warmup):
        evaluate()
    sync()
    warmup_seconds = time.perf_counter()-started if warmup else 0.
    times, first_values, repeat_delta = [], None, 0.
    for _ in range(repeats):
        sync()
        started = time.perf_counter()
        prediction = evaluate()
        sync()
        times.append(time.perf_counter()-started)
        values = finite_array(prediction.moments.detach().cpu().numpy(), (10,), "predicted moments")
        if first_values is None:
            first_values = values.copy()
        else:
            repeat_delta = max(repeat_delta, float(np.max(np.abs(values-first_values))))
    started = time.perf_counter()
    physics = _physical_metrics(values, case, residual_target)
    physics_seconds = time.perf_counter()-started
    return dict(
        full_cell_rule=rule, h=blocks.pitch, pitch=blocks.pitch, h_over_pitch=1.,
        status="AVAILABLE", operation=operand.operation, moments=values,
        moment_error=_moment_metrics(values, case["native_moments"], scale),
        derivative_error=_derivative_metrics(prediction, case, scale), physical_residual_error=physics,
        physical_metric_seconds=physics_seconds, model_weights_used=model_weights_used, device=device,
        actual_evaluated_band_nodes=prediction.diagnostics.get("band_nodes"),
        actual_active_nodes=prediction.diagnostics.get("active_nodes"),
        potential_material_nodes=material.node_count_for_rule(rule) if material is not None else None,
        packed_default_nodes=int(len(blocks.nodes)), diagnostics=prediction.diagnostics,
        query_seconds=times, query_seconds_median=float(np.median(times)),
        query_seconds_min=float(min(times)), warmup_queries=warmup, warmup_seconds=warmup_seconds,
        measured_queries=repeats, repeated_moment_max_abs_difference=repeat_delta,
        query_timing_scope=("model.evaluate_operand with pre-encoded geometry, including query preparation/cache and synchronized device work; excludes encoder, CPU metric/export and checkpoint loading"
                            if model_weights_used else
                            "evaluate_packed(model=None); excludes surface setup, input load, material audit and metrics"))


def compare_case(case, *, rules=RULES, chunk_size=65536, repeats=1, warmup=0,
                 residual_target=None, material_audit=None, frozen_model=None):
    """Predict all rules at one unchanged query, with a common native target.

    residual_target, when present, must come from the strict manifest sidecar
    loader. Binding, reference moments, multiplier conventions and profile are
    checked again here. Direct case CLI inputs do not accept unbound profiles.
    """
    rules = _options(rules, chunk_size, repeats, warmup)
    scale = validate_case(case)
    _validate_bound_target(case, residual_target)
    blocks, operand, meta = case["blocks"], case["operand"], case["meta"]
    material = getattr(blocks, "material", None)
    native = finite_array(case["native_moments"], (10,), "native moments")
    started = time.perf_counter()
    if material_audit is None:
        material_audit = _audit_blocks(blocks, scale, case.get("material_cache_moments"), chunk_size)
        audit_seconds = time.perf_counter() - started
        audit_source = "checked_for_this_case"
    else:
        audit_seconds, audit_source = 0., "reused_same_loaded_material_audit"
    full_error = (np.asarray(material_audit["cubature_moments"])-case["native_full_editable_moments"]) / scale
    full_max = float(np.max(np.abs(full_error)))
    result = dict(schema=CASE_SCHEMA, status="COMPARISON_COMPLETE_NOT_NATIVE_ACCEPTANCE",
                  reference_kind=meta["reference_kind"], provenance=meta,
                  asset_receipts=case.get("asset_receipts"), manifest_receipt=case.get("manifest_receipt"),
                  material_identity=blocks.identity,
                  material_source=material.source if material is not None else "UNAVAILABLE_PACKED_PROVENANCE",
                  material_asset=case.get("asset_receipts", {}).get("material"),
                  partial_cell_rule=material.partial_rule if material is not None else None,
                  full_cell_count=material.full_cell_count if material is not None else None,
                  partial_cell_count=material.partial_cell_count if material is not None else None,
                  fixed_grid=dict(side=blocks.side, origin=blocks.origin, pitch=blocks.pitch,
                                  h=blocks.pitch, h_over_pitch=1., block_voxels=blocks.block_voxels),
                  raw_coefficients=case["coefficients"], fixed_moments=case["fixed_moments"],
                  native_moments=native, native_full_editable_moments=case["native_full_editable_moments"],
                  fixed_residual_reference=residual_target,
                  moment_scale=scale, model_weights_used=False, parameter_updates=0,
                  native_solver_acceptance="NOT_EVALUATED", original_five_percent_gate="UNCHANGED_NOT_EVALUATED",
                  native_integral_seconds=None, native_integral_timing_status="NOT_MEASURED_EXISTING_LABELS_ONLY",
                  load_and_pack_seconds=case.get("load_and_pack_seconds"),
                  material_audit=material_audit, material_audit_seconds=audit_seconds,
                  material_audit_source=audit_source,
                  full_material_reference_scaled_max=full_max,
                  full_material_reference_status="PASS" if full_max <= 1e-8 else "FAIL",
                  rules=list(rules), records=[], learned_rules=[],
                  learned_comparison_status="NOT_REQUESTED" if frozen_model is None else "REQUESTED_NOT_RUN")
    if full_max > 1e-8:
        result.update(status="STOP_FULL_MATERIAL_REFERENCE",
                      reason="full editable ten-moment error exceeds unchanged physical-scale tolerance 1e-8")
        return jsonable(result)
    response_module = import_module(PREFIX + ".v21_response")
    evaluate_packed = import_module(PREFIX + ".compact_pino_v21").evaluate_packed
    started = time.perf_counter()
    surface = response_module.prepare_surface(operand.vertices, operand.faces)
    result["surface_setup_seconds"] = time.perf_counter() - started
    velocity = operand.vertex_velocity if meta["derivative_valid"] else None
    for rule in rules:
        def evaluate():
            return evaluate_packed(blocks, surface, blocks.pitch, operand.operation,
                                   case["fixed_moments"], velocity, model=None,
                                   chunk_size=chunk_size, full_cell_rule=rule)
        result["records"].append(_measure_rule(evaluate, case, scale, rule, repeats=repeats,
            warmup=warmup, residual_target=residual_target))
    if frozen_model is not None:
        learned_started = time.perf_counter()
        encoded, encoding = frozen_model.encode(case)
        result["learned_geometry_encoding"] = encoding
        for rule in rules:
            def evaluate_learned():
                return frozen_model.predict(encoded, case, rule, chunk_size)
            row = _measure_rule(evaluate_learned, case, scale, rule, repeats=repeats,
                warmup=warmup, residual_target=residual_target, synchronize=frozen_model.synchronize,
                model_weights_used=True, device=str(frozen_model.device))
            row["checkpoint_sha256"] = frozen_model.receipt["sha256"]
            result["learned_rules"].append(row)
        frozen_model.assert_frozen()
        result.update(model_weights_used=True, learned_comparison_status="FROZEN_PAIRED_COMPARISON_COMPLETE",
                      learned_case_seconds=time.perf_counter()-learned_started,
                      learned_case_timing_scope="geometry cache lookup/encoding plus all learned warmups, query forwards, CPU exports and metrics; checkpoint preparation excluded")
    if operand_fingerprint(operand) != meta["operand_content_sha256"] or query_fingerprint(case["coefficients"]) != meta["query_content_sha256"]:
        raise ValueError("operand or raw coefficients changed during the common-query comparison")
    return jsonable(result)


def compare_records(records, *, rules=RULES, chunk_size=65536, repeats=1, warmup=0, source=None,
                    frozen_model=None):
    """Compare an explicit list without pooling mismatched or missing queries."""
    rules = _options(rules, chunk_size, repeats, warmup)
    if not records:
        raise ValueError("at least one eligible frozen case is required")
    if len({r["record_id"] for r in records}) != len(records):
        raise ValueError("record IDs must be unique")
    audits, rows = {}, []
    started = time.perf_counter()
    for record in records:
        case = record["case"]
        meta = case["meta"]
        if (record.get("case_sha256") is not None
                and record["case_sha256"] != case.get("manifest_receipt", {}).get("sha256")):
            raise ValueError("record case SHA256 differs from its loaded frozen case")
        key = (id(case["blocks"]), meta["editable_mass"], meta["length_scale"])
        item = compare_case(case, rules=rules, chunk_size=chunk_size, repeats=repeats, warmup=warmup,
                            residual_target=record.get("residual_target"), material_audit=audits.get(key),
                            frozen_model=frozen_model)
        audits[key] = item["material_audit"]
        item.update(record_id=record["record_id"], role=record.get("role", meta["split"]),
                    qualification_receipt=record.get("qualification_receipt"),
                    residual_target_receipt=record.get("residual_target_receipt"))
        rows.append(item)
    complete = all(row["status"] == "COMPARISON_COMPLETE_NOT_NATIVE_ACCEPTANCE" for row in rows)
    operator = import_module(PREFIX + ".compact_pino_v21")
    import torch
    modules = [import_module(PREFIX + suffix) for suffix in
               (".compact_pino_v21", ".v21_material", ".v21_blocks", ".v21_response", ".v21_response_surface")]
    sources = {str(Path(module.__file__).resolve()): sha256(module.__file__) for module in modules}
    sources[str(Path(__file__).resolve())] = sha256(__file__)
    frozen_snapshot = None if frozen_model is None else frozen_model.snapshot()
    if frozen_model is not None:
        from ..training.runner import _execution_sources
        root = Path(__file__).resolve().parents[3]
        sources.update({str(root/name): digest for name, digest in _execution_sources(frozen_model.model).items()})
        checkpoint_source = Path(__file__).with_name("frozen_checkpoint.py")
        sources[str(checkpoint_source.resolve())] = sha256(checkpoint_source)
    return jsonable(dict(
        schema=SCHEMA, status="COMPARISON_COMPLETE_NOT_NATIVE_ACCEPTANCE" if complete else "INCOMPLETE_MATERIAL_REFERENCE",
        model_weights_used=any(row["model_weights_used"] for row in rows), parameter_updates=0, source=source,
        frozen_model=frozen_snapshot,
        native_solver_acceptance="NOT_EVALUATED", original_five_percent_gate="UNCHANGED_NOT_EVALUATED",
        source_roles=sorted({row["role"] for row in rows}),
        selected_record_ids=[row["record_id"] for row in rows],
        rules=list(rules), bandwidth_policy="h equals unchanged material pitch for every rule",
        timing_scope="evaluate_packed only: includes classification, rule expansion, response and reduction; excludes load, packing, audit, surface setup and metrics",
        timing_order="case order; declared no-learning rule order then declared learned rule order; no randomized benchmark or speedup claim",
        learned_timing_scope="checkpoint preparation, encoder/cache lookup, synchronized evaluate_operand query forward, and case end-to-end seconds are separate; pure retention kernels are not separately profiled",
        chunk_size=chunk_size, warmup_queries_per_rule=warmup, measured_queries_per_rule=repeats,
        elapsed_seconds=time.perf_counter()-started,
        runtime=dict(python=platform.python_version(), numpy=np.__version__, torch=torch.__version__,
                     device="cpu", device_scope="no_learning_baseline",
                     learned_device=None if frozen_model is None else str(frozen_model.device),
                     torch_threads=torch.get_num_threads(), operator_module=operator.__name__),
        execution_sources=sources, cases=rows))


def _direct_cases(paths, cache_directory):
    records, static_cache, verification_cache = [], {}, {}
    for path in paths:
        case = load_baseline_case(path, cache_directory=cache_directory, static_cache=static_cache,
                                  asset_verification_cache=verification_cache)
        qualification = None
        if case["meta"]["reference_kind"] == "deployed_native":
            receipt = case["asset_receipts"].get("source_qualification")
            if receipt is None:
                raise ValueError("real direct cases need their exported teacher qualification; prefer --manifest")
            declaration = json.loads(Path(receipt["path"]).read_text())
            record_id = declaration["record_id"]
            qualification = verify_export_qualification(case, record_id,
                                                        asset_verification_cache=verification_cache)
        else:
            record_id = case["meta"]["geometry_id"] + "/" + case["meta"]["query_id"]
        records.append(dict(record_id=record_id, role=case["meta"]["split"], case=case,
                            qualification_receipt=qualification))
    return records


def _load_cli_inputs(args):
    started = time.perf_counter()
    if args.manifest:
        from ..training.manifest import load_training_manifest
        data = load_training_manifest(args.manifest, mode=args.mode, cache_directory=args.cache_directory)
        records = [r for r in data.records if r["loader_eligible"] and
                   (not args.roles or r["role"] in args.roles)]
        selected_ids = {r["record_id"] for r in records}
        source = dict(kind="strict_export_manifest", path=str(data.path), sha256=data.manifest_sha256,
                      snapshot_id=data.snapshot_id, mode=args.mode, source_admission=data.source_admission,
                      selected_roles=args.roles or ["train", "development"],
                      excluded_record_ids=[r["record_id"] for r in data.records if r["record_id"] not in selected_ids])
    else:
        records = _direct_cases(args.case, args.cache_directory)
        source = dict(kind="direct_frozen_cases", source_authority="NOT_RECHECKED_WITHOUT_MANIFEST",
                      profile_status="UNAVAILABLE_PROFILE_USE_BOUND_MANIFEST_SIDECAR")
    return records, source, time.perf_counter()-started


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    inputs = parser.add_mutually_exclusive_group(required=True)
    inputs.add_argument("--manifest", help="existing frozen export training manifest; strict loader reused without training")
    inputs.add_argument("--case", action="append", help="existing frozen case JSON; repeat for multiple common queries")
    parser.add_argument("--mode", choices=("smoke", "trial"), default="trial",
                        help="strict manifest loading mode; smoke requires explicitly synthetic cases")
    parser.add_argument("--roles", nargs="+", choices=("train", "development"),
                        help="optional manifest-only role filter; selected IDs are recorded")
    parser.add_argument("--rules", nargs="+", choices=RULES, default=list(RULES))
    parser.add_argument("--chunk-size", type=int, default=65536)
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--warmup", type=int, default=0)
    parser.add_argument("--cache-directory")
    parser.add_argument("--checkpoint", help="optional current-runner checkpoint for paired frozen learned_rules")
    parser.add_argument("--checkpoint-sha256", help="required exact lowercase SHA256 with --checkpoint")
    parser.add_argument("--device", default="cpu", help="checkpoint inference device: cpu or cuda[:index]; no-learning stays CPU")
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    _options(args.rules, args.chunk_size, args.repeats, args.warmup)
    output = Path(args.output)
    if output.exists():
        parser.error("output must be new; existing diagnostic reports are never overwritten")
    if args.case and args.roles:
        parser.error("--roles only applies to a frozen manifest")
    if bool(args.checkpoint) != bool(args.checkpoint_sha256):
        parser.error("--checkpoint and --checkpoint-sha256 must be supplied together")
    if not args.checkpoint and args.device != "cpu":
        parser.error("--device only selects optional checkpoint inference; no-learning runs on CPU")
    frozen_model = None
    if args.checkpoint:
        from .frozen_checkpoint import load_frozen_checkpoint
        frozen_model = load_frozen_checkpoint(args.checkpoint, args.checkpoint_sha256, device=args.device)
    try:
        records, source, load_seconds = _load_cli_inputs(args)
        result = compare_records(records, rules=args.rules, chunk_size=args.chunk_size,
                                 repeats=args.repeats, warmup=args.warmup, source=source,
                                 frozen_model=frozen_model)
    finally:
        if frozen_model is not None:
            frozen_model.close()
    result["input_load_seconds"] = load_seconds
    with output.open("x", encoding="utf8") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps(dict(output=str(output.resolve()), status=result["status"],
                          cases=len(result["cases"]), model_weights_used=result["model_weights_used"],
                          native_solver_acceptance="NOT_EVALUATED")))
    return 0 if result["status"] == "COMPARISON_COMPLETE_NOT_NATIVE_ACCEPTANCE" else 2


if __name__ == "__main__":
    raise SystemExit(main())
