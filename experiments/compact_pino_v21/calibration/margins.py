"""Score the actual strict native line-search margins and perturb its recorded bordered system."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from capture import PROTOCOL, save, sha


def dual_coordinates(q):
    q = np.asarray(q, float)
    t = np.diag([1.0] * 7 + [0.5] * 3)
    s = np.linalg.solve(t, q)
    s /= np.linalg.norm(s)
    g = (np.eye(10) - np.outer(q, q)) @ t / np.linalg.norm(t @ s)
    return s, g


def geodesic(s, d, step):
    t = d - s * s.dot(d)
    t /= np.linalg.norm(t)
    p = np.cos(step) * s + np.sin(step) * t
    p[7:] *= 0.5
    return p / np.linalg.norm(p)


def kkt_score(q, m):
    # Native static maximum-mass mode: F=-m0, C=[mx,my]. The
    # dual-sphere multiplier solve is independent of the predicted moments.
    s, _ = dual_coordinates(q)
    eq = np.eye(10)[:, [1, 2]]
    grad = -np.eye(10)[0]
    P = np.column_stack((-eq, s))
    params = np.linalg.lstsq(P, -grad, rcond=None)[0]
    optimality = grad + P @ params
    return (
        np.linalg.norm(np.r_[optimality, m[1:3]]),
        params[-1],
        np.dot(optimality, optimality),
    )


def residual_interval(m, stationarity_squared, budget):
    a = np.abs(np.asarray(m)[1:3])
    lower = np.maximum(a - budget, 0.0)
    return [
        np.sqrt(stationarity_squared + lower @ lower),
        np.sqrt(stationarity_squared + (a + budget) @ (a + budget)),
    ]


def direction(system, rhs, s):
    solved = np.linalg.solve(system, rhs)[:10]
    return solved - s * np.dot(s, solved)


def proposals(base, trial, system_info, bound, length, mass):
    payload = system_info["newton_system_payload"]
    A = np.array(payload["matrix"])
    r = np.array(payload["right_hand_side"])
    p = base["evaluation_payload"]
    q = np.array(base["quadric_coefficients"])
    s, G = dual_coordinates(q)
    m = np.array(p["solver_raw_moments"])
    coefficient = np.linalg.norm(q)
    delta = bound * mass * length / coefficient
    if A.shape != (13, 13):
        raise ValueError("unexpected static native bordered matrix")
    d = direction(A, r, s)
    h = np.linalg.norm(d)
    max_step = min(h, 0.5)
    delta_raw = G @ d
    u = delta_raw / np.linalg.norm(delta_raw)
    if abs(u @ q) > 1e-12:
        raise ValueError("input is not tangent at base")
    step = float(trial["step"])
    q1 = np.array(trial["quadric_coefficients"])
    native_q1 = geodesic(s, d, step)
    mismatch = float(np.max(np.abs(q1 - native_q1)))
    if mismatch > 1e-7:
        raise ValueError(f"native trial reproduction discrepancy {mismatch}")
    out = []
    for s1, s2 in ((1, 1), (1, -1), (-1, 1), (-1, -1)):
        # The two first-moment components realize this Jacobian at the base with
        # bounded smooth functions e_i(q)=sign*0.01*scale*sin(5*u·q_hat).
        # Each component has |e| <= .01 scale and ||q|| ||de/dq|| <= .05 scale.
        dJ = np.zeros((10, 10))
        dJ[1] = s1 * delta * u
        dJ[2] = s2 * delta * u
        Aa = A.copy()
        Aa[10:12, :10] += dJ[[1, 2]] @ G
        try:
            da = direction(Aa, r, s)
        except np.linalg.LinAlgError:
            out.append(dict(signs=[s1, s2], singular=True))
            continue
        other_step = min(0.5, np.linalg.norm(da)) * step / max_step
        qa = geodesic(s, da, other_step)
        out.append(
            dict(
                signs=[s1, s2],
                q_trial=qa.tolist(),
                initial_step=float(min(0.5, np.linalg.norm(da))),
                q_trial_delta=float(np.linalg.norm(qa - q1)),
                direction_change_degrees=float(
                    np.degrees(
                        np.arccos(
                            np.clip(np.dot(d, da) / (h * np.linalg.norm(da)), -1, 1)
                        )
                    )
                ),
            )
        )
    return dict(
        raw_tangent=u.tolist(),
        q_base=q.tolist(),
        M_base=m.tolist(),
        baseline_reproduction_max_abs=mismatch,
        direction_bound=float(bound),
        baseline_initial_step=float(max_step),
        perturbed=out,
    )


def assess(path):
    captured = json.loads(Path(path).read_text())
    policy = json.loads(PROTOCOL.read_text())
    bound = policy["moment_component_limit"]
    jb = policy["jacobian_direction_limit"]
    assert (
        captured["split"] == "train" and captured["geometry"] in policy["train_assets"]
    )
    if captured["mode"] != "STATIC_BALANCE" or captured["objective"] != "MAXIMUM_MASS":
        raise ValueError("different native moment problem")
    length = captured["length"]
    mass = captured["full_editable"][0]
    margins = []
    linear = None
    base = None
    maximum_error = 0.0
    initial = None
    accepted = 0
    rejected = 0
    for e in captured["events"]:
        if e["type"] == "newton_linear_solve":
            linear = e
            continue
        if e["type"] != "evaluation":
            continue
        if not e.get("evaluation_payload") or e.get("raw_equation_residual") is None:
            if e["iteration"] > 0:
                margins.append(
                    dict(
                        iteration=e["iteration"],
                        step=e["step"],
                        outcome=e["outcome"],
                        reason=e["reason"],
                        geometry_rejected=True,
                        detail=e["detail"],
                    )
                )
            continue
        p = e["evaluation_payload"]
        m = np.array(p["solver_raw_moments"])
        score, mu, stationary = kkt_score(e["quadric_coefficients"], m)
        error = abs(score - e["raw_equation_residual"])
        maximum_error = max(maximum_error, error)
        if error > 2e-12 or abs(mu - e["multiplier_mu"]) > 2e-10:
            raise ValueError("failed native merit/multiplier reconstruction")
        limits = residual_interval(m, stationary, bound * mass * length)
        if e["iteration"] == 0:
            base = (e, limits)
            initial = dict(raw=score, mu=mu, reason=e["reason"])
            continue
        if base is None:
            raise ValueError("missing native accepted state")
        b, bl = base
        expected = score < b["raw_equation_residual"]
        is_accept = e["outcome"] == "accepted"
        if (
            e["reason"] not in ["stationary_multiplier_nonpositive"]
            and expected != is_accept
        ):
            raise ValueError("native decrease rule mismatch")
        margin = b["raw_equation_residual"] - score
        lower = bl[0] - limits[1]
        upper = bl[1] - limits[0]
        row = dict(
            iteration=e["iteration"],
            step=e["step"],
            outcome=e["outcome"],
            reason=e["reason"],
            q_native=e["quadric_coefficients"],
            native_margin=margin,
            fixed_trial_moment_margin_interval=[lower, upper],
            base_score=b["raw_equation_residual"],
            trial_score=score,
            positive_decrease_certified_under_moment_box=bool(lower > 0),
            nondecrease_certified_under_moment_box=bool(upper <= 0),
            native_score_decision_flip_possible_under_independent_moment_errors=bool(
                lower <= 0 < upper
            ),
        )
        if linear and linear["iteration"] == e["iteration"]:
            row["jacobian_sensitivity"] = proposals(
                b, e, linear["event"] if "event" in linear else linear, jb, length, mass
            )
        margins.append(row)
        if is_accept:
            accepted += 1
            base = (e, limits)
        else:
            rejected += 1
    r = captured.get("result", {})
    return dict(
        geometry=captured["geometry"],
        start=captured["start"],
        seed=captured["seed"],
        mode=captured["mode"],
        capture_sha256=sha(path),
        status=r.get("status"),
        termination_reason=next(
            (
                e["reason"]
                for e in reversed(captured["events"])
                if e["type"] == "attempt_terminal"
            ),
            None,
        ),
        function_evaluations=r.get("function_evaluations"),
        native_trace_hash=r.get("trace_hash"),
        native_event_hash=r.get("event_hash"),
        evaluator_identity=captured["evaluator_identity"],
        initial=initial,
        scored_native_accepts=accepted,
        scored_native_backtracks=rejected,
        max_native_score_reproduction_error=maximum_error,
        decisions=margins,
    )


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("captures", nargs="+", type=Path)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    save(
        args.output,
        dict(
            protocol_sha256=sha(PROTOCOL),
            analysis_sha256=sha(Path(__file__)),
            scope="fixed training starts and conditional perturbations",
            rule="Strict native raw Eq.14 residual decrease; topology transitions are scored after a fresh original Boolean. Scale = editable mass times native bounding-box longest extent for first moments.",
            counterfactual="Moment intervals rigorously enclose each fixed-base/trial residual for independent per-component bounds. Jacobian perturbations change the native proposed trial; the computed new coordinate alone has no measured material score.",
            prefixes=[assess(path) for path in args.captures],
        ),
    )


if __name__ == "__main__":
    main()
