"""Paired full-model efficiency benchmark, with numerical checks before timing.

The default is the actual Token16 model on two analytic fixture queries.  The
reference disables all five execution optimizations; both arms load exactly the
same weights.  No optimizer step, native solve, accuracy promotion, or qualified
solution speedup is claimed.  A real manifest is accepted only through the
existing ``trial`` source/teacher admission loader.
"""
from __future__ import annotations

import argparse
from datetime import datetime
import gc
import hashlib
import inspect
import json
import math
import os
from pathlib import Path
import platform
import sys
import time
from zoneinfo import ZoneInfo

import numpy as np
import torch

from research.NeuralMoments.models.pcraft_ufno.candidate_projects.quadrann_reproduction.spin_adapter.compact_pino_v21 import CompactPINOv21
from research.NeuralMoments.models.pcraft_ufno.candidate_projects.quadrann_reproduction.spin_adapter.v22_inference_cache import InferenceGeometryCache, geometry_feature_key
from ..p0_diagnostics.baseline_cases import jsonable, moment_scale, sha256
from ..training.losses import physical_loss
from ..training.manifest import load_training_manifest
from ..training.runner import CHECKPOINT_SCHEMA, _execution_sources
from ..training.smoke import make_smoke_manifest
from ..training.state import seed_everything


SCHEMA = "tvcg.v22.operator-efficiency-benchmark.v1"
# Frozen acceptance criteria: compare computation, not prediction vs a teacher.
TOLERANCES = dict(moment_max_scaled_absolute=1e-7,
                  jacobian_max_scaled_absolute=1e-6,
                  parameter_gradient_rtol=2e-4,
                  parameter_gradient_atol=1e-9)


def _write(path, value):
    with Path(path).open("x", encoding="utf8") as stream:
        json.dump(jsonable(value), stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")


def _tensor_digest(state):
    digest = hashlib.sha256()
    for name, value in sorted(state.items()):
        digest.update(name.encode("utf8"))
        if value is None:
            digest.update(b"none")
        else:
            array = value.detach().cpu().contiguous().numpy()
            digest.update(repr((array.dtype.str, array.shape)).encode("ascii"))
            digest.update(array.tobytes())
    return digest.hexdigest()


def _source_fingerprints(model):
    root = Path(__file__).resolve().parents[3]
    result = _execution_sources(model)
    paths = [Path(__file__), Path(__file__).with_name("__init__.py")]
    adapter = root/"research/NeuralMoments/models/pcraft_ufno/candidate_projects/quadrann_reproduction/spin_adapter"
    paths.extend(adapter/name for name in ("v22_geometry_inputs.py", "v22_query_cache.py", "v22_inference_cache.py"))
    for module in model.modules():
        source = inspect.getsourcefile(type(module))
        if source is not None and Path(source).resolve().is_relative_to(root):
            paths.append(Path(source).resolve())
    for path in paths:
        result[str(path.relative_to(root))] = sha256(path)
    return dict(sorted(result.items()))


def _synchronize(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _release(model, device):
    model.zero_grad(set_to_none=True)
    model.clear_efficiency_caches()
    # Moving the only live arm off CUDA also removes its learned parameter
    # storage from the next arm's peak-allocation accounting.
    if device.type == "cuda":
        model.cpu()
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()


def _new_model(state, *, efficiency, token_grid, device, static_cache_bytes, query_cache_bytes):
    model = CompactPINOv21(token_grid=token_grid, efficiency=efficiency,
                          static_cache_bytes=static_cache_bytes, query_cache_bytes=query_cache_bytes)
    model.load_state_dict(state, strict=True)
    model.to(device)
    return model


def _one_prediction(model, record, *, training, chunk_size):
    case = record["case"]
    features = model.encode_geometry(case["blocks"], case["fixed_moments"],
                                     density=case["meta"]["density"], inference=not training)
    return model.evaluate_operand(features, case["operand"], with_jacobian=True, chunk_size=chunk_size)


def _train_forward_backward(model, record, chunk_size):
    model.train()
    model.zero_grad(set_to_none=True)
    prediction = _one_prediction(model, record, training=True, chunk_size=chunk_size)
    loss, info = physical_loss([prediction], [record], jacobian_loss_weight=1.)
    if not info["moment_denominator"] or not info["jacobian_denominator"]:
        raise ValueError("benchmark requires an actual positive Moment and Jacobian loss denominator")
    if not loss.requires_grad or not torch.isfinite(loss):
        raise ValueError("benchmark training example must have a finite parameter-dependent joint loss")
    loss.backward()
    return prediction, loss, info


def _gradient_groups(gradients):
    groups = {}
    for label, prefix in (("encoder", "shared_encoder.pino_encoder."),
                          ("geometry_attention", "geometry_attention."),
                          ("response", "response.")):
        values = [value for name, value in gradients.items() if name.startswith(prefix) and value is not None]
        finite = bool(values) and all(bool(torch.isfinite(value).all()) for value in values)
        norm = math.sqrt(sum(float(value.abs().double().square().sum()) for value in values)) if finite else None
        groups[label] = dict(tensors=len(values), finite=finite, norm=norm,
                             finite_nonzero=bool(finite and norm > 0.))
    return groups


def _capture_validation(model, records, *, chunk_size, seed):
    model.eval()
    rows = []
    with torch.no_grad():
        for record in records:
            prediction = _one_prediction(model, record, training=False, chunk_size=chunk_size)
            if prediction.jacobian is None:
                raise ValueError("benchmark query has no regular response Jacobian: " + record["record_id"])
            if not bool(torch.isfinite(prediction.moments).all()) or not bool(torch.isfinite(prediction.jacobian).all()):
                raise FloatingPointError("nonfinite numerical benchmark response")
            rows.append(dict(record_id=record["record_id"], moments=prediction.moments.detach().cpu().numpy(),
                             jacobian=prediction.jacobian.detach().cpu().numpy(), diagnostics=prediction.diagnostics))
    seed_everything(seed)
    _, loss, info = _train_forward_backward(model, records[0], chunk_size)
    gradients = {name: None if value.grad is None else value.grad.detach().cpu().clone()
                 for name, value in model.named_parameters()}
    summary = dict(rows=rows, training_record_id=records[0]["record_id"], joint_loss=float(loss.detach()),
                   loss_membership=info, gradient_groups=_gradient_groups(gradients),
                   gradient_sha256=_tensor_digest(gradients))
    return summary, gradients


def _compare(reference, candidate, reference_gradients, candidate_gradients, records):
    rows = []
    for expected, observed, record in zip(reference["rows"], candidate["rows"], records):
        case = record["case"]
        scale = moment_scale(case["meta"]["editable_mass"], case["meta"]["length_scale"])
        dm = np.abs(observed["moments"]-expected["moments"])/scale
        dj = np.abs(observed["jacobian"]-expected["jacobian"])*case["parameter_scales"][None, :]/scale[:, None]
        rows.append(dict(record_id=record["record_id"], moment_max_scaled_absolute=float(dm.max()),
                         jacobian_max_scaled_absolute=float(dj.max()),
                         moment_pass=bool(dm.max() <= TOLERANCES["moment_max_scaled_absolute"]),
                         jacobian_pass=bool(dj.max() <= TOLERANCES["jacobian_max_scaled_absolute"])))
    if reference_gradients.keys() != candidate_gradients.keys():
        raise AssertionError("optimized model changed parameter names")
    gradient_rows, worst_absolute, worst_fraction, numerator, denominator = [], 0., 0., 0., 0.
    mismatched_none = []
    for name, expected in reference_gradients.items():
        observed = candidate_gradients[name]
        if expected is None or observed is None:
            if (expected is None) != (observed is None):
                mismatched_none.append(name)
            continue
        if not bool(torch.isfinite(expected).all()) or not bool(torch.isfinite(observed).all()):
            raise FloatingPointError("nonfinite parameter gradient: " + name)
        delta = (observed-expected).abs().double()
        bound = TOLERANCES["parameter_gradient_atol"] + TOLERANCES["parameter_gradient_rtol"]*expected.abs().double()
        absolute = float(delta.max()) if delta.numel() else 0.
        fraction = float((delta/bound).max()) if delta.numel() else 0.
        gradient_rows.append(dict(name=name, max_absolute_difference=absolute,
                                  max_tolerance_fraction=fraction, passed=fraction <= 1.))
        worst_absolute, worst_fraction = max(worst_absolute, absolute), max(worst_fraction, fraction)
        numerator += float(delta.square().sum())
        denominator += float(expected.abs().double().square().sum())
    gradients_pass = not mismatched_none and worst_fraction <= 1.
    groups_pass = all(group["finite_nonzero"] for arm in (reference, candidate) for group in arm["gradient_groups"].values())
    passed = all(row["moment_pass"] and row["jacobian_pass"] for row in rows) and gradients_pass and groups_pass
    return dict(status="PASS" if passed else "FAIL", tolerances=dict(TOLERANCES), queries=rows,
                parameter_gradients=dict(passed=gradients_pass, compared_tensors=len(gradient_rows),
                    none_membership_mismatches=mismatched_none, max_absolute_difference=worst_absolute,
                    max_tolerance_fraction=worst_fraction,
                    relative_l2_difference=math.sqrt(numerator/denominator) if denominator else None,
                    per_parameter=gradient_rows), nonzero_encoder_attention_response_gradients=groups_pass)


def _evaluation_sequence(model, records, *, chunk_size, feature_cache_bytes):
    model.eval()
    cache = InferenceGeometryCache(max_bytes=feature_cache_bytes)
    outputs = []
    with torch.no_grad():
        for record in records:
            case = record["case"]
            if model.efficiency:
                key = geometry_feature_key(case["blocks"], case["fixed_moments"], case["meta"]["density"])
                features = cache.get(key)
                if features is None:
                    features = model.encode_geometry(case["blocks"], case["fixed_moments"],
                        density=case["meta"]["density"], inference=True)
                    cache[key] = features
                prediction = model.evaluate_operand(features, case["operand"], with_jacobian=True, chunk_size=chunk_size)
            else:
                prediction = _one_prediction(model, record, training=False, chunk_size=chunk_size)
            outputs.append(prediction)
    stats = cache.snapshot()
    cache.clear()
    return outputs, stats


def _timed(function, device):
    _synchronize(device)
    memory = None
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
        memory = dict(allocated_before_bytes=torch.cuda.memory_allocated(device))
    start = time.perf_counter()
    result = function()
    _synchronize(device)
    seconds = time.perf_counter()-start
    if memory is not None:
        memory.update(peak_allocated_bytes=torch.cuda.max_memory_allocated(device),
                      allocated_after_bytes=torch.cuda.memory_allocated(device))
        memory["peak_above_start_bytes"] = memory["peak_allocated_bytes"]-memory["allocated_before_bytes"]
    return result, dict(seconds=seconds, cuda_memory=memory)


def _summarize(samples):
    times = [sample["seconds"] for sample in samples]
    return dict(samples=samples, repeats=len(samples), median_seconds=float(np.median(times)),
                p95_seconds=float(np.percentile(times, 95)),
                p95_scope="descriptive interpolation of a small repeated microbenchmark, not a population tail guarantee")


def _clear(model):
    model.geometry_input_cache.clear(reset_stats=True)
    model.query_preparation_cache.clear(reset_stats=True)


def _measure_arm(model, records, *, device, repeats, warmup, query_repetitions,
                 chunk_size, feature_cache_bytes, seed):
    # Warm runtime/Numba and the actual neural path before the cache-cold case.
    model.eval()
    for _ in range(warmup):
        with torch.no_grad():
            _one_prediction(model, records[0], training=False, chunk_size=chunk_size)
    cold = []
    for _ in range(repeats):
        _clear(model)
        seed_everything(seed)
        def query():
            with torch.no_grad():
                return _one_prediction(model, records[0], training=False, chunk_size=chunk_size)
        result, sample = _timed(query, device)
        sample["cache_after"] = model.efficiency_statistics()
        cold.append(sample)
        del result
    sequence = records*query_repetitions
    _clear(model)
    for _ in range(warmup):
        _evaluation_sequence(model, sequence, chunk_size=chunk_size, feature_cache_bytes=feature_cache_bytes)
    evaluation = []
    for _ in range(repeats):
        seed_everything(seed)
        result, sample = _timed(lambda: _evaluation_sequence(model, sequence, chunk_size=chunk_size,
                                                              feature_cache_bytes=feature_cache_bytes), device)
        sample["learned_feature_cache"] = result[1]
        sample["cache_after"] = model.efficiency_statistics()
        sample["queries"] = len(sequence)
        sample["seconds_per_query"] = sample["seconds"]/len(sequence)
        evaluation.append(sample)
        del result
    _clear(model)
    for _ in range(warmup):
        seed_everything(seed)
        _train_forward_backward(model, records[0], chunk_size)
    training = []
    for _ in range(repeats):
        seed_everything(seed)
        model.zero_grad(set_to_none=True)
        result, sample = _timed(lambda: _train_forward_backward(model, records[0], chunk_size), device)
        sample["joint_loss"] = float(result[1].detach())
        sample["cache_after"] = model.efficiency_statistics()
        training.append(sample)
        del result
    return dict(
        cache_cold_single_query=dict(**_summarize(cold),
            boundary="model loaded and runtime/Numba warm; static/query caches empty each repeat; includes encoder inputs, neural encoding, finite-surface preparation and M+J; excludes mesh preprocessing/admission/model loading"),
        frozen_query_evaluation=dict(**_summarize(evaluation),
            boundary="static/query caches warm; fresh learned-feature cache per sequence; reference encodes every query; same two frozen queries repeated, not a Newton trajectory",
            query_ids=[row["record_id"] for row in sequence]),
        joint_forward_backward=dict(**_summarize(training),
            boundary="batch=1; static/query caches warm; learned geometry recomputed every pass; forward, normalized joint loss and backward; no optimizer step or checkpoint I/O",
            record_id=records[0]["record_id"]))


def _select_records(data, record_ids):
    eligible = [row for row in data.train if row["case"]["meta"]["derivative_valid"]
                and row["moment_weight"] > 0 and row["jacobian_weight"] > 0]
    if record_ids:
        if len(record_ids) != 2 or len(set(record_ids)) != 2:
            raise ValueError("exactly two distinct --record-id values are required")
        by_id = {row["record_id"]: row for row in eligible}
        if any(key not in by_id for key in record_ids):
            raise ValueError("benchmark record IDs must be eligible training rows with qualified positive-weight J")
        selected = [by_id[key] for key in record_ids]
    else:
        grouped = {}
        selected = None
        for row in eligible:
            case = row["case"]
            key = geometry_feature_key(case["blocks"], case["fixed_moments"], case["meta"]["density"])
            grouped.setdefault(key, []).append(row)
            if len(grouped[key]) == 2:
                selected = grouped[key]
                break
        if selected is None:
            raise ValueError("benchmark needs two qualified training queries sharing one static geometry/task context")
    keys = [geometry_feature_key(row["case"]["blocks"], row["case"]["fixed_moments"],
                                 row["case"]["meta"]["density"]) for row in selected]
    if keys[0] != keys[1]:
        raise ValueError("the paired benchmark queries must share geometry, fixed material and density")
    return selected


def run_benchmark(output, *, manifest=None, mode="smoke", device="cpu", repeats=3, warmup=1,
                  query_repetitions=4, token_grid=16, seed=830251, chunk_size=65536,
                  static_cache_bytes=128*1024**2, query_cache_bytes=128*1024**2,
                  feature_cache_bytes=64*1024**2, record_ids=None, checkpoint=None,
                  checkpoint_sha256=None):
    if mode not in ("smoke", "trial"):
        raise ValueError("mode must be smoke or trial")
    for name, value in (("repeats", repeats), ("warmup", warmup), ("query_repetitions", query_repetitions),
                        ("chunk_size", chunk_size)):
        if type(value) is not int or value < 1:
            raise ValueError(name + " must be a positive integer")
    if token_grid not in (8, 16) or type(seed) is not int:
        raise ValueError("Token8/Token16 and an integer seed required")
    for name, value in (("static_cache_bytes", static_cache_bytes), ("query_cache_bytes", query_cache_bytes),
                        ("feature_cache_bytes", feature_cache_bytes)):
        if type(value) is not int or value < 0:
            raise ValueError(name + " must be a nonnegative integer")
    device = torch.device(device)
    if device.type not in ("cpu", "cuda"):
        raise ValueError("benchmark supports CPU or CUDA")
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise ValueError("CUDA requested but unavailable")
        if os.environ.get("CUBLAS_WORKSPACE_CONFIG") not in (":4096:8", ":16:8"):
            raise ValueError("set CUBLAS_WORKSPACE_CONFIG=:4096:8 before launching deterministic CUDA benchmark")
    if bool(checkpoint) != bool(checkpoint_sha256):
        raise ValueError("a checkpoint and its explicit SHA256 must be provided together")
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    if manifest is None:
        if mode != "smoke":
            raise ValueError("a frozen admitted manifest is required for real-data benchmarking")
        manifest = make_smoke_manifest(output/"assets")
    data = load_training_manifest(manifest, mode=mode, cache_directory=output/"material-cache")
    records = _select_records(data, record_ids)
    seed_everything(seed)
    torch.backends.cudnn.allow_tf32 = False
    initial = CompactPINOv21(token_grid=token_grid, efficiency=False)
    initial_source = dict(kind="seeded_initialization", seed=seed)
    if checkpoint:
        path = Path(checkpoint).resolve(strict=True)
        if sha256(path) != checkpoint_sha256:
            raise ValueError("checkpoint SHA256 mismatch")
        saved = torch.load(path, map_location="cpu", weights_only=True)
        if saved.get("schema") != CHECKPOINT_SCHEMA or saved.get("config", {}).get("token_grid") != token_grid:
            raise ValueError("only this runner's checkpoint schema with matching token_grid is supported")
        initial.load_state_dict(saved["model"], strict=True)
        initial_source = dict(kind="runner_checkpoint_model_only", path=str(path), sha256=checkpoint_sha256)
    state = {name: value.detach().cpu().clone() for name, value in initial.state_dict().items()}
    fingerprints = _source_fingerprints(initial)
    parameter_count = sum(parameter.numel() for parameter in initial.parameters())
    del initial
    gc.collect()
    report = dict(schema=SCHEMA, created_at=datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(),
        scope="synthetic_software_only" if mode == "smoke" else "admitted_training_queries_software_only",
        status="RUNNING", tolerances=dict(TOLERANCES), initial_weights=initial_source,
        initial_tensor_sha256=_tensor_digest(state), source_sha256=fingerprints,
        configuration=dict(token_grid=token_grid, seed=seed, chunk_size=chunk_size, repeats=repeats,
            warmup=warmup, query_repetitions=query_repetitions, static_cache_bytes=static_cache_bytes,
            query_cache_bytes=query_cache_bytes, feature_cache_bytes=feature_cache_bytes,
            parameter_count=parameter_count, encoder="unchanged default shared P-CRAFT/U-FNO 32+64",
            measured_arm_order=["reference", "optimized"], optimizer_updates=0),
        data=dict(manifest_path=str(data.path), manifest_sha256=data.manifest_sha256,
            snapshot_id=data.snapshot_id, loader_scope=data.scope, source_admission=data.source_admission,
            selected_training_queries=[dict(record_id=row["record_id"], case_sha256=row["case_sha256"],
                geometry_id=row["case"]["meta"]["geometry_id"], query_id=row["case"]["meta"]["query_id"],
                derivative_dimension=row["case"]["meta"]["derivative_dimension"]) for row in records]),
        runtime=dict(python=sys.version, platform=platform.platform(), numpy=np.__version__,
            torch=torch.__version__, cuda_runtime=torch.version.cuda, device=str(device),
            cuda_name=torch.cuda.get_device_name(device) if device.type == "cuda" else None,
            torch_threads=torch.get_num_threads(), torch_interop_threads=torch.get_num_interop_threads(),
            deterministic_algorithms=torch.are_deterministic_algorithms_enabled(),
            cublas_workspace_config=os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
            cuda_matmul_tf32=torch.backends.cuda.matmul.allow_tf32, cudnn_tf32=torch.backends.cudnn.allow_tf32,
            thread_environment={name:os.environ.get(name) for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")}),
        interpretation=dict(native_physical_acceptance="NOT_EVALUATED", trained_model_quality="NOT_EVALUATED",
            qualified_solution_speedup_claim=False, long_training_authorized=False,
            cpu_memory_scope="retained cache payload only; no process-RSS or CPU activation-peak claim",
            timing_statistics="small-sample descriptive medians/P95; no speed threshold, confidence interval or population generalization"))
    factory = dict(token_grid=token_grid, device=device, static_cache_bytes=static_cache_bytes,
                   query_cache_bytes=query_cache_bytes)
    model = None
    try:
        validation = {}
        gradients = {}
        for arm, enabled in (("reference", False), ("optimized", True)):
            model = _new_model(state, efficiency=enabled, **factory)
            validation[arm], gradients[arm] = _capture_validation(model, records, chunk_size=chunk_size, seed=seed)
            _release(model, device)
            del model
            model = None
        report["validation"] = validation
        report["equivalence"] = _compare(validation["reference"], validation["optimized"],
                                           gradients["reference"], gradients["optimized"], records)
        del gradients
        if report["equivalence"]["status"] != "PASS":
            raise AssertionError("numerical/gradient equivalence failed; timings not run")
        report["measurements"] = {}
        for arm, enabled in (("reference", False), ("optimized", True)):
            model = _new_model(state, efficiency=enabled, **factory)
            report["measurements"][arm] = _measure_arm(model, records, device=device, repeats=repeats,
                warmup=warmup, query_repetitions=query_repetitions, chunk_size=chunk_size,
                feature_cache_bytes=feature_cache_bytes, seed=seed)
            final_sources = _source_fingerprints(model)
            _release(model, device)
            del model
            model = None
        report["comparison"] = {}
        for name, baseline in report["measurements"]["reference"].items():
            candidate = report["measurements"]["optimized"][name]
            ratio = candidate["median_seconds"]/baseline["median_seconds"]
            report["comparison"][name] = dict(optimized_over_reference_median_ratio=ratio,
                reference_over_optimized_median_speed_ratio=1./ratio,
                median_time_change_percent=(ratio-1.)*100.,
                scope="measured invocation only; no native-solver speedup inference")
        report["execution_sources_unchanged"] = final_sources == fingerprints
        if not report["execution_sources_unchanged"]:
            report["source_sha256_after"] = final_sources
            raise RuntimeError("execution sources changed during benchmark; rerun after source freeze")
        report["status"] = "PASS"
    except Exception as exc:
        report["status"] = "FAIL"
        report["failure"] = dict(type=type(exc).__name__, message=str(exc))
        _write(output/"benchmark_report.json", report)
        raise
    finally:
        if model is not None:
            _release(model, device)
    _write(output/"benchmark_report.json", report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, help="new directory for immutable benchmark evidence")
    parser.add_argument("--manifest", help="frozen training manifest; omitted only to create synthetic smoke fixtures")
    parser.add_argument("--mode", choices=("smoke", "trial"), default="smoke")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--query-repetitions", type=int, default=4)
    parser.add_argument("--token-grid", type=int, choices=(8, 16), default=16)
    parser.add_argument("--seed", type=int, default=830251)
    parser.add_argument("--chunk-size", type=int, default=65536)
    parser.add_argument("--static-cache-bytes", type=int, default=128*1024**2)
    parser.add_argument("--query-cache-bytes", type=int, default=128*1024**2)
    parser.add_argument("--feature-cache-bytes", type=int, default=64*1024**2)
    parser.add_argument("--record-id", action="append", dest="record_ids")
    parser.add_argument("--checkpoint")
    parser.add_argument("--checkpoint-sha256")
    args = parser.parse_args()
    result = run_benchmark(**vars(args))
    print(json.dumps(dict(status=result["status"], scope=result["scope"],
                         equivalence=result["equivalence"]["status"], comparison=result["comparison"],
                         report=str(Path(args.output).resolve()/"benchmark_report.json")), indent=2))


if __name__ == "__main__":
    main()
