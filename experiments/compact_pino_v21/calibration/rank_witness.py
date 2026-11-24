"""Exercise the original linear solver with an in-budget ranked Jacobian perturbation."""

import argparse
import json
from pathlib import Path

import numpy as np

from capture import DEFAULT_SOLVER, PROTOCOL, registered_case, save, sha


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--capture", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--native-root", type=Path, default=Path("/tmp/tvcg-v21-native"))
    p.add_argument("--solver-registration", type=Path, default=Path(DEFAULT_SOLVER))
    args = p.parse_args()
    baseline = json.loads(args.capture.read_text())
    policy = json.loads(PROTOCOL.read_text())
    assert (
        baseline["geometry"] == "S001"
        and baseline["query_id"] == "S001_registered_ellipsoid"
    )
    req, spin, meta, backend, files, length = registered_case(
        "S001",
        args.native_root,
        args.solver_registration,
        policy,
        policy["solver_prefix"],
    )
    native = baseline["calls"][0]
    q0 = np.asarray(native["q"])
    M = np.asarray(native["M"])
    J = np.asarray(native["J"])
    S = length * meta["mesh_full_editable"][0]
    # A power-of-two ratio is represented exactly in binary64. This lets the
    # production solver actually observe the dependent constraint rows.
    J1 = J.copy()
    J1[2] = -0.5 * J1[1]
    change = J1[2] - J[2]
    normalized_J_error = float(np.linalg.norm(change) * np.linalg.norm(q0) / S)
    if normalized_J_error >= policy["jacobian_direction_limit"]:
        raise ValueError("witness exceeds the declared budget")
    initial = next(
        e
        for e in baseline["events"]
        if e["type"] == "evaluation" and e["iteration"] == 0
    )
    payload = initial["evaluation_payload"]
    seq = []
    calls = []

    def fn(q):
        q = np.asarray(q)
        calls.append(q.tolist())
        if np.array_equal(q, q0):
            # The exact base packet was measured by the unmodified original
            # production map and is authenticated by the entire capture SHA.
            return dict(
                raw_moments=M,
                variable_raw_moments=M - np.asarray(meta["fixed_moments"]),
                moment_jacobian=J1,
                derivative_defined=True,
                geometry_hash=native["geometry_hash"],
                topology_hash=native["topology_hash"],
                evaluation_map_identity=payload["evaluation_map_identity"]
                + "+S001-bounded-sensitivity",
                boolean_backend_identity=payload["boolean_backend_identity"],
                derivative_identity="bounded-training-perturbation",
            )
        out = backend(q)
        raw = q - q0
        amp = policy["moment_component_limit"] * S
        angle = (raw @ change) / amp
        m = np.asarray(out["raw_moments"]).copy()
        jac = np.asarray(out["moment_jacobian"]).copy()
        m[2] += amp * np.sin(angle)
        jac[2] += np.cos(angle) * change
        out.update(
            raw_moments=m,
            variable_raw_moments=m - np.asarray(meta["fixed_moments"]),
            moment_jacobian=jac,
            evaluation_map_identity=payload["evaluation_map_identity"]
            + "+S001-bounded-sensitivity",
            derivative_identity="bounded-training-perturbation",
        )
        backend._materials.clear()
        return out

    res = spin.solve_spin_it_faster_with_external_moment_evaluator(
        req, fn, initial_quadrics=baseline["explicit_guesses"], event_sink=seq.append
    )
    assert calls and np.array_equal(calls[0], q0)
    before = next(e for e in baseline["events"] if e["type"] == "newton_linear_solve")[
        "newton_system_payload"
    ]
    after = next(e for e in seq if e.get("newton_system_payload"))[
        "newton_system_payload"
    ]
    beforeA, afterA = np.array(before["matrix"]), np.array(after["matrix"])
    files.update({str(path): sha(path) for path in [Path(__file__), args.capture]})
    out = dict(
        geometry="S001",
        start=baseline["query_id"],
        seed=0,
        split="train",
        files=files,
        evaluator=meta["evaluator_identity"],
        scope="Counterfactual sensitivity experiment at a measured training state. Only the candidate moment map was perturbed. The compiler, seed, actual starting geometry and native strict line-search policy are identical.",
        norm_domain="The compiled solver normalizes its raw query. An explicit smooth field on unit raw inputs is e_y(q)=0.01*S*sin((q-q0) dot deltaJ_y/(0.01*S)); other deviations are zero. Its exact base moment error is zero. On every unit raw input its componentwise value is <=1%, and its maximal directional derivative is bounded by the reported row norm. The radial derivative at this base is recorded separately.",
        q_base=q0,
        M_base=M,
        raw_row_perturbation=change,
        scale=S,
        global_M_component_bound=policy["moment_component_limit"],
        global_J_direction_bound_on_unit_queries=normalized_J_error,
        base_radial_leak_scaled=abs(change @ q0) / S,
        base_M_error=0.0,
        base_J_row_relation="J_y = -0.5 J_x",
        original_first_trial=next(
            e["reason"]
            for e in baseline["events"]
            if e["type"] == "evaluation" and e["iteration"] == 1
        ),
        original_ranks=[before["computed_rank"], before["equilibrated_rank"]],
        counterfactual_ranks=[after["computed_rank"], after["equilibrated_rank"]],
        original_bordered_system=beforeA,
        perturbed_bordered_system=afterA,
        exact_bordered_row_dependence=bool(
            np.array_equal(afterA[11], -0.5 * afterA[10])
        ),
        native_counterfactual_status=res["status"],
        native_diagnostic=res["diagnostic"],
        native_events=[
            dict(type=e["type"], reason=e["reason"], outcome=e["outcome"]) for e in seq
        ],
        original_event_hash=baseline["result"]["event_hash"],
        modified_event_hash=res["event_hash"],
        modified_function_evaluations=res["function_evaluations"],
    )
    save(args.output, out)
    print(
        out["original_ranks"],
        out["counterfactual_ranks"],
        out["native_counterfactual_status"],
        out["native_diagnostic"],
        normalized_J_error,
        flush=True,
    )


if __name__ == "__main__":
    main()
