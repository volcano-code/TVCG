"""Re-evaluate bounded derivative-perturbed Newton proposals with the true training oracle."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

from capture import DEFAULT_SOLVER, PROTOCOL, registered_case, save, sha
from margins import assess, kkt_score


def smooth_field(q, base_q, direction, signs, scale, M_budget, J_budget):
    q = np.asarray(q, dtype=float)
    norm = np.linalg.norm(q)
    unit = q / norm
    frequency = J_budget / M_budget
    argument = frequency * np.dot(direction, unit - base_q)
    amplitude = M_budget * scale * np.asarray(signs)
    delta = np.zeros(10)
    jac = np.zeros((10, 10))
    delta[1:3] = amplitude * np.sin(argument)
    du = direction - unit * np.dot(direction, unit)
    jac[1:3] = amplitude[:, None] * frequency * np.cos(argument) * du[None, :] / norm
    return delta, jac


def native(backend, q):
    try:
        x = backend(q)
        return dict(
            M=np.asarray(x["raw_moments"]),
            J=np.asarray(x["moment_jacobian"]),
            geometry_hash=x["geometry_hash"],
            topology_hash=x["topology_hash"],
        )
    except (ValueError, RuntimeError) as exc:
        return dict(error=str(exc))
    finally:
        backend._materials.clear()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--native-root", type=Path, default=Path("/tmp/tvcg-v21-native"))
    p.add_argument("--solver-registration", type=Path, default=Path(DEFAULT_SOLVER))
    p.add_argument("--capture", type=Path, required=True)
    p.add_argument("--iteration", type=int, choices=[1, 2], required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    policy = json.loads(PROTOCOL.read_text())
    source = assess(args.capture)
    assert source["geometry"] == "T2" and source["start"] == "registered"
    row = next(
        r
        for r in source["decisions"]
        if r["iteration"] == args.iteration and r["outcome"] == "accepted"
    )
    sen = row["jacobian_sensitivity"]
    req, spin, meta, backend, files, length = registered_case(
        "T2",
        args.native_root,
        args.solver_registration,
        policy,
        policy["solver_prefix"],
    )
    files.update(
        {
            str(p): sha(p)
            for p in [
                Path(__file__),
                args.capture,
                Path(sys.modules["margins"].__file__),
            ]
        }
    )
    mass = meta["mesh_full_editable"][0]
    scale = mass * length
    u = np.array(sen["raw_tangent"])
    q0 = np.array(sen["q_base"])
    b = np.array(sen["M_base"])
    mb = policy["moment_component_limit"]
    jb = policy["jacobian_direction_limit"]
    base_score, mu, _ = kkt_score(q0, b)
    out = dict(
        training_geometry="T2",
        split="train",
        evaluation="counterfactual train oracle for altered proposal, separate from original native trajectory",
        files=files,
        profile=meta["profile"],
        evaluator_identity=backend.context_identity,
        native_iteration=args.iteration,
        q0=q0,
        u=u,
        M0=b,
        base_score=base_score,
        unperturbed_true_margin=row["native_margin"],
        unperturbed_was_accepted=True,
        scale=scale,
        M_component_bound=mb,
        J_row_operator_bound=jb,
        proposals=[],
    )
    for alt in sen["perturbed"]:
        qa = np.array(alt["q_trial"])
        s = alt["signs"]
        e0, j0 = smooth_field(q0, q0, u, s, scale, mb, jb)
        e, j = smooth_field(qa, q0, u, s, scale, mb, jb)
        np.testing.assert_allclose(e0, 0.0, atol=1e-15)
        np.testing.assert_allclose(
            np.linalg.norm(j0, axis=1)[1:3] * np.linalg.norm(q0) / scale, jb, atol=1e-13
        )
        value = native(backend, qa)
        detail = {**alt, "material": value}
        if "error" not in value:
            m = np.asarray(value["M"])
            true, mu1, _ = kkt_score(qa, m)
            altered, mu2, _ = kkt_score(qa, m + e)
            detail.update(
                true_margin=base_score - true,
                perturbed_approximate_margin=base_score - altered,
                mu_true=mu1,
                true_trial_decreases=bool(true < base_score),
                perturbed_score_decreases=bool(altered < base_score),
                moment_error_scaled=float(np.max(np.abs(e)) / scale),
                direction_error_scaled=float(
                    np.max(np.linalg.norm(j, axis=1)) * np.linalg.norm(qa) / scale
                ),
            )
        out["proposals"].append(detail)
        save(args.output, out)
        print(
            args.iteration,
            s,
            detail.get("true_margin"),
            detail.get("perturbed_approximate_margin"),
            value.get("error"),
            flush=True,
        )
    if args.iteration == 1:
        out["native_directional_fd"] = []
        for h in (1e-4, 5e-5):
            samples = [native(backend, q0 + sign * h * u) for sign in (-1, 1)]
            fd = dict(
                step=h,
                M_minus=samples[0].get("M"),
                M_plus=samples[1].get("M"),
                errors=[t.get("error") for t in samples],
            )
            if all("M" in t for t in samples):
                initial = next(
                    e
                    for e in json.loads(args.capture.read_text())["calls"]
                    if np.array_equal(e["q"], q0)
                )
                j = np.array(initial["J"]) @ u
                other = (np.array(samples[1]["M"]) - samples[0]["M"]) / (2 * h)
                scales = mass * length ** np.array([0, 1, 1, 1, 2, 2, 2, 2, 2, 2])
                fd.update(
                    Jv=j,
                    centered=other,
                    max_scaled_directional_error=np.max(np.abs(other - j) / scales),
                    endpoint_topology_hashes=[s["topology_hash"] for s in samples],
                    base_topology_hash=initial["topology_hash"],
                )
            out["native_directional_fd"].append(fd)
            save(args.output, out)
    out["complete"] = True
    save(args.output, out)


if __name__ == "__main__":
    main()
