"""Replay-gated, fixed-query Moment/Jacobian attribution.

The adapter delegates acceptance and one-step updates to the frozen deployment
solver. This module supplies no substitute optimum or Newton formula. Every arm
starts with a fresh copy of the same EXTERNAL state; endogenous multipliers
follow the frozen solver's own policy for the supplied moments.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import importlib
import inspect
import json
from pathlib import Path

import numpy as np

ORDER = ("1", "x", "y", "z", "xx", "yy", "zz", "yz", "xz", "xy")
DEGREE = np.array([0, 1, 1, 1, 2, 2, 2, 2, 2, 2])
COEFFICIENT_MULTIPLIERS = (1., 1., 1., 1., 1., 1., 1., 2., 2., 2.)
ACCEPT_KEYS = ("stationarity_residual", "balance_residual", "accepted")


def moment_scale(editable_mass, length):
    if not np.isfinite([editable_mass, length]).all() or min(editable_mass, length) <= 0:
        raise ValueError("positive editable mass and length required")
    result = editable_mass * length ** DEGREE
    if not np.isfinite(result).all() or np.any(result <= 0):
        raise ValueError("non-finite physical moment scales")
    return result


def finite_array(x, shape, name):
    a = np.asarray(x, dtype=np.float64)
    if a.shape != shape or not np.isfinite(a).all():
        raise ValueError(f"{name} must be finite with shape {shape}")
    return a.copy()


def sha256(path):
    with open(path, "rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def jsonable(value):
    if isinstance(value, np.ndarray):
        return jsonable(value.tolist())
    if isinstance(value, np.generic):
        return jsonable(value.item())
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(v) for v in value]
    if isinstance(value, float) and not np.isfinite(value):
        raise ValueError("non-finite report value; use an explicit status and null")
    return value


def _hash(value, name):
    if not isinstance(value, str) or len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
        raise ValueError(f"invalid {name} SHA-256")
    return value


def _nonnegative(value, name):
    if isinstance(value, bool) or not isinstance(value, (float, int)) or not np.isfinite(value) or value < 0:
        raise ValueError(f"{name} must be finite and nonnegative")
    return float(value)


def verify_provenance(meta):
    for name in ("geometry_id", "query_id", "source_group_id", "teacher_version", "solver_version",
                 "coordinate_frame", "units", "chart_id", "trajectory_stage"):
        if not isinstance(meta.get(name), str) or not meta[name].strip():
            raise ValueError(f"missing {name}")
    # These diagnostics are part of development; sealed tests must not enter it.
    if meta.get("split") not in ("train", "development", "diagnostic", "synthetic"):
        raise ValueError("explicit non-blind development split required")
    for name in ("geometry", "operand", "manifest", "checkpoint", "solver"):
        _hash(meta.get(name + "_sha256"), name)
    if tuple(meta.get("moment_order", ())) != ORDER or meta.get("coefficient_basis") != "raw_coefficients":
        raise ValueError("moment order or raw coefficient convention mismatch")
    if tuple(meta.get("coefficient_multipliers", ())) != COEFFICIENT_MULTIPLIERS:
        raise ValueError("native doubled-cross-term coefficient convention required")
    th = meta.get("thresholds", {})
    for name in ("stationarity", "balance"):
        if _nonnegative(th.get(name), name + " threshold") <= 0:
            raise ValueError("actual positive frozen thresholds required")
    contract = meta.get("adapter_contract", {})
    if type(contract.get("acceptance_reads_jacobian")) is not bool:
        raise ValueError("declare whether actual acceptance reads the cut Jacobian")
    if contract.get("multiplier_policy") not in ("native_recompute", "native_frozen"):
        raise ValueError("declare actual native multiplier policy")
    _hash(contract.get("external_state_sha256"), "external solver state")
    tolerances = meta.get("replay_tolerances", {})
    for key in ("residual_atol", "residual_rtol", "direction_atol", "direction_rtol",
                "coefficient_atol", "coefficient_rtol"):
        _nonnegative(tolerances.get(key), "replay " + key)
    if type(meta.get("derivative_valid")) is not bool:
        raise ValueError("derivative_valid must be an explicit boolean")
    if not meta["derivative_valid"] and not str(meta.get("derivative_reason", "")).strip():
        raise ValueError("unavailable derivative needs its reason; zero is not a mask")


def verify_artifacts(meta, directory):
    """Verify bytes supplied to a real CLI run, not just the shape of hash strings.

    The adapter remains responsible for loading THESE assets and reproducing
    the query. Matching hashes alone cannot certify an implementation's semantics.
    """
    directory = Path(directory)
    artifacts = meta.get("artifacts", {})
    receipt = {}
    for role in ("geometry", "operand", "manifest", "checkpoint", "solver", "solver_state"):
        item = artifacts.get(role)
        if not isinstance(item, dict) or not isinstance(item.get("path"), str) or not item["path"]:
            raise ValueError(f"missing actual {role} artifact path")
        path = Path(item["path"])
        path = path if path.is_absolute() else directory / path
        path = path.resolve(strict=True)
        actual = sha256(path)
        expected = (meta["adapter_contract"]["external_state_sha256"] if role == "solver_state"
                    else meta[role + "_sha256"])
        if actual != expected:
            raise ValueError(f"{role} artifact checksum mismatch")
        receipt[role] = dict(path=str(path), sha256=actual, bytes=path.stat().st_size)
    return receipt


def _record(result, meta, dimension, *, step=False):
    if not isinstance(result, dict):
        raise TypeError("native adapter must return a dict")
    out = jsonable(result)
    if not step:
        for key in ACCEPT_KEYS[:2]:
            _nonnegative(out.get(key), key)
        if type(out.get("accepted")) is not bool:
            raise ValueError("native acceptance must return a boolean")
        if out["accepted"] and any(out[key] > meta["thresholds"][name] for key, name in
                                   zip(ACCEPT_KEYS[:2], ("stationarity", "balance"))):
            raise ValueError("native acceptance contradicts frozen thresholds")
    else:
        if type(out.get("line_search_accepted")) is not bool:
            raise ValueError("step must return boolean line_search_accepted")
        if out.get("direction") is not None:
            finite_array(out["direction"], (dimension,), "direction in the fixed chart")
        elif out["line_search_accepted"]:
            raise ValueError("accepted step has no native direction")
        if not isinstance(out.get("reason"), str) or not out["reason"].strip():
            raise ValueError("step must report the native reason")
        if out.get("trial_coefficients") is not None:
            finite_array(out["trial_coefficients"], (10,), "actual trial coefficients")
        if out["line_search_accepted"] and out.get("trial_coefficients") is None:
            raise ValueError("accepted step has no actual trial")
        for key in ("trial_native_stationarity_residual", "trial_native_balance_residual"):
            if out.get(key) is not None:
                _nonnegative(out[key], key)
            elif out["line_search_accepted"]:
                raise ValueError("accepted step has no independent native trial residual")
    return out


def _replay_matches(actual, reference, meta, *, step=False):
    if not isinstance(reference, dict):
        raise ValueError("missing archived native replay reference")
    tol = meta["replay_tolerances"]
    checked = []
    if not step:
        groups = [(key, "residual") for key in ACCEPT_KEYS[:2]]
        flags = ["accepted"]
    else:
        groups = [("direction", "direction"), ("trial_coefficients", "coefficient"),
                  ("trial_native_stationarity_residual", "residual"),
                  ("trial_native_balance_residual", "residual")]
        flags = ["line_search_accepted", "reason"]
    for key in flags:
        if key not in reference or actual[key] != reference[key]:
            raise ValueError(f"native replay mismatch: {key}")
        checked.append(key)
    for key, unit in groups:
        if key not in reference:
            raise ValueError(f"native replay reference is missing {key}")
        left, right = actual.get(key), reference[key]
        if left is None or right is None:
            if left is not None or right is not None:
                raise ValueError(f"native replay mismatch: {key}")
        else:
            left, right = np.asarray(left), np.asarray(right)
            if left.shape != right.shape or not np.isfinite(right).all() or not np.allclose(
                    left, right, atol=tol[unit + "_atol"], rtol=tol[unit + "_rtol"]):
                raise ValueError(f"native replay mismatch: {key}")
        checked.append(key)
    return checked


def _direction_comparison(candidate, native, basis):
    result = {}
    for name, x, y in (("chart", candidate, native),
                       ("raw_coefficient", basis @ candidate, basis @ native)):
        nx, ny = np.linalg.norm(x), np.linalg.norm(y)
        delta = float(np.linalg.norm(x - y))
        result[name] = dict(absolute_l2=delta, relative_l2=delta / ny if ny else None,
                            cosine=float(np.clip(x @ y / (nx * ny), -1, 1)) if nx * ny else None)
    return result


def attribution(case, factory, *, run_step=False):
    """Run native replay first; use fresh external state for every substitution.

    Optional no_learning_moments / initial_moments are evaluated at this exact
    query. Missing alternative Jacobians remain null. Mixed M/J arms diagnose
    local sensitivity; they are not coherent deployable operator functions.
    """
    meta = copy.deepcopy(case["meta"])
    verify_provenance(meta)
    c = finite_array(case["coefficients"], (10,), "coefficients")
    p = finite_array(case["predicted_moments"], (10,), "predicted moments")
    n = finite_array(case["native_moments"], (10,), "native moments")
    scale = moment_scale(float(meta["editable_mass"]), float(meta["length_scale"]))
    valid = meta["derivative_valid"]
    pj = nj = basis = None
    d = None
    if valid:
        d = meta.get("derivative_dimension")
        if type(d) is not int or d not in (3, 8, 10):
            raise ValueError("derivative dimension must be 3, 8 or 10")
        pj = finite_array(case["predicted_jacobian"], (10, d), "predicted Jacobian")
        nj = finite_array(case["native_jacobian"], (10, d), "native Jacobian")
        basis = finite_array(case["chart_basis"], (10, d), "raw chart basis")
        # This is chart qualification, not an assumed rank of the physical J.
        if np.linalg.matrix_rank(basis) != d:
            raise ValueError("chart basis is rank deficient")
        if d == 10 and not np.array_equal(basis, np.eye(10)):
            raise ValueError("Full J must be expressed in raw coefficients")
    elif run_step:
        raise ValueError("one-step comparison requires matched qualified derivatives")
    elif meta["adapter_contract"]["acceptance_reads_jacobian"]:
        raise ValueError("this native acceptance requires an unavailable Jacobian")

    def evaluate(m, j, *, step=False):
        adapter = factory(copy.deepcopy(case))
        # The native implementation owns the multiplier policy, objective,
        # constraints, chart, linear solve, and trial evaluation/line search.
        fn = adapter.step if step else adapter.accept
        return _record(fn(c.copy(), m.copy(), None if j is None else j.copy()), meta, d, step=step)

    reference = meta.get("native_replay", {})
    native_accept = evaluate(n, nj)
    replay = {"accept": _replay_matches(native_accept, reference.get("accept"), meta)}
    native_step = None
    if run_step:
        native_step = evaluate(n, nj, step=True)
        replay["step"] = _replay_matches(native_step, reference.get("step"), meta, step=True)

    arms = [("A_predM_predJ", p, pj), ("B_nativeM_predJ", n, pj)]
    if valid:
        arms.extend([("C_predM_nativeJ", p, nj), ("D_nativeM_nativeJ", n, nj)])
    fixed = {name: native_accept if name == "D_nativeM_nativeJ" else evaluate(m, j)
             for name, m, j in arms}
    base = fixed["A_predM_predJ"]
    # When an archived prediction-side acceptance exists, it is an independent
    # second anchor for M-dependent native multiplier recomputation.
    if meta.get("predicted_replay") is not None:
        replay["predicted_accept"] = _replay_matches(base, meta["predicted_replay"], meta)
    repeated = evaluate(p, pj)
    _replay_matches(repeated, base, meta)
    invariance = None
    if valid:
        invariance = True
        for a, b in (("A_predM_predJ", "C_predM_nativeJ"), ("B_nativeM_predJ", "D_nativeM_nativeJ")):
            try:
                _replay_matches(fixed[a], fixed[b], meta)
            except ValueError:
                invariance = False
        if not meta["adapter_contract"]["acceptance_reads_jacobian"] and not invariance:
            raise ValueError("acceptance declared Moment-only but changes when only J changes")

    single = []
    for i, name in enumerate(ORDER):
        m = p.copy()
        m[i] = n[i]
        result = evaluate(m, pj)
        single.append(dict(component=name, predicted=p[i], native=n[i],
                           scaled_error=(p[i] - n[i]) / scale[i], acceptance=result,
                           stationarity_delta=result["stationarity_residual"] - base["stationarity_residual"]))
    alternatives = {}
    for label in ("no_learning", "initial"):
        key = label + "_moments"
        if key not in case:
            continue
        m = finite_array(case[key], (10,), key)
        j = None
        if valid and label + "_jacobian" in case:
            if meta.get(label + "_derivative_valid") is not True:
                raise ValueError(f"explicit qualification required for {label} Jacobian")
            j = finite_array(case[label + "_jacobian"], (10, d), label + " Jacobian")
        if j is None and meta["adapter_contract"]["acceptance_reads_jacobian"]:
            accept = None
        else:
            accept = evaluate(m, j)
        alternatives[label] = dict(moments=m, scaled_moment_errors=(m - n) / scale,
                                    fixed_endpoint=accept, derivative_status="matched_valid" if j is not None else "unavailable",
                                    one_step=evaluate(m, j, step=True) if run_step and j is not None else None)

    report = dict(schema="tvcg.v22.endpoint-attribution.v2", provenance=meta, coefficients=c,
                  status="DIAGNOSTIC_ONLY_NO_TRAINING_PROMOTION", native_replay_verified=replay,
                  replay_scope="numerical_callback_reproduction_not_independent_query_asset_binding",
                  native_physical_acceptance_validated=False,
                  native_fixed_endpoint=native_accept, fixed_endpoint=fixed,
                  alternative_responses=alternatives, single_component_replacements=single,
                  single_component_interpretation="sensitivity_not_additive_causal_contributions",
                  scaled_moment_errors=(p - n) / scale,
                  max_scaled_moment_error=float(np.max(np.abs(p - n) / scale)),
                  derivative_status="matched_valid" if valid else "unavailable_no_jacobian_claim",
                  endpoint_J_invariance_observed=invariance, fixed_endpoint_repeat=repeated,
                  fixed_endpoint_repeat_identical=all(repeated[k] == base[k] for k in ACCEPT_KEYS),
                  fixed_endpoint_repeat_within_frozen_tolerance=True,
                  one_step=None)
    if valid and case.get("direction") is not None:
        v = finite_array(case["direction"], (d,), "actual trajectory direction")
        raw_norm = np.linalg.norm(basis @ v)
        if raw_norm == 0:
            report["directional_derivative"] = dict(status="zero_direction_not_informative",
                direction=v, raw_direction=basis @ v, predicted_Jv=pj @ v, native_Jv=nj @ v,
                scaled_actual_direction_error=((pj - nj) @ v) / scale, scaled_unit_direction_error=None)
        else:
            coefficient_scale = _nonnegative(meta.get("coefficient_scale"), "coefficient scale")
            if coefficient_scale == 0:
                raise ValueError("positive frozen coefficient scale required")
            unit = v * coefficient_scale / raw_norm
            report["directional_derivative"] = dict(status="available", direction=v, raw_direction=basis @ v,
                predicted_Jv=pj @ v, native_Jv=nj @ v, scaled_actual_direction_error=((pj - nj) @ v) / scale,
                coefficient_scale=coefficient_scale, scaled_unit_direction_error=((pj - nj) @ unit) / scale)
    elif valid:
        report["directional_derivative"] = dict(status="direction_unavailable")
    if run_step:
        report["one_step"] = {name: native_step if name == "D_nativeM_nativeJ" else evaluate(m, j, step=True)
                              for name, m, j in arms}
        native_direction = native_step.get("direction")
        report["step_direction_comparison"] = {name: _direction_comparison(
            np.asarray(arm["direction"]), np.asarray(native_direction), basis)
            if arm.get("direction") is not None and native_direction is not None else None
            for name, arm in report["one_step"].items()}
        # A new adapter object can still share a global backend or mutable cache.
        # Repeat at the end as well as replaying the archived native step first.
        repeats = {"D_nativeM_nativeJ": evaluate(n, nj, step=True),
                   "A_predM_predJ": evaluate(p, pj, step=True)}
        for name, repeated_step in repeats.items():
            _replay_matches(repeated_step, report["one_step"][name], meta, step=True)
        report["one_step_repeats"] = repeats
        report["one_step_repeat_within_frozen_tolerance"] = True
    native_repeat = evaluate(n, nj)
    _replay_matches(native_repeat, native_accept, meta)
    report["native_fixed_endpoint_repeat"] = native_repeat
    return jsonable(report)


def load_factory(name):
    module, sep, attr = name.partition(":")
    if not sep or not module or not attr:
        raise ValueError("adapter must be module:factory")
    factory = getattr(importlib.import_module(module), attr)
    if not callable(factory):
        raise ValueError("adapter factory is not callable")
    return factory


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", required=True, type=Path, help="NPZ with same-query numeric arrays")
    parser.add_argument("--meta", required=True, type=Path, help="frozen identity, replay and solver contract")
    parser.add_argument("--adapter", required=True, help="actual native module:factory")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--asset-root", type=Path, help="relative artifact base; defaults to metadata parent")
    parser.add_argument("--one-step", action="store_true")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("refusing to overwrite an existing audit")
    meta = json.loads(args.meta.read_text(encoding="utf8"))
    verify_provenance(meta)
    receipts = verify_artifacts(meta, args.asset_root or args.meta.parent)
    with np.load(args.case, allow_pickle=False) as data:
        case = dict(data)
    case["meta"] = meta
    # Adapters use these resolved, byte-verified paths, independent of their cwd.
    case["verified_artifacts"] = receipts
    case["asset_root"] = str((args.asset_root or args.meta.parent).resolve())
    factory = load_factory(args.adapter)
    try:
        source = inspect.getsourcefile(factory)
    except TypeError:
        source = None
    adapter_receipt = dict(entrypoint=args.adapter, source_path=None, sha256=None)
    if source is not None:
        source = Path(source).resolve(strict=True)
        adapter_receipt.update(source_path=str(source), sha256=sha256(source))
    if meta.get("adapter_sha256") is not None and adapter_receipt["sha256"] != _hash(meta["adapter_sha256"], "adapter"):
        raise ValueError("native adapter source checksum mismatch")
    report = attribution(case, factory, run_step=args.one_step)
    report["adapter_source"] = adapter_receipt
    report["verified_artifacts"] = receipts
    report["input_hashes"] = dict(case=sha256(args.case), meta=sha256(args.meta))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf8") as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
        stream.write("\n")


if __name__ == "__main__":
    main()
