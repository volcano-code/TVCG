"""Capture original native Newton decisions on registered training material only."""

from __future__ import annotations

import argparse
from dataclasses import replace
import hashlib
import inspect
import json
from pathlib import Path
import sys
from time import perf_counter

import numpy as np

HERE = Path(__file__).resolve().parent
PROTOCOL = HERE / "protocol.json"
DEFAULT_SOLVER = "/data/xvxinyao/TVCG-phase2-12-operator/runs/phase3_1b/solver"


def sha(path):
    with Path(path).open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def clean(x):
    if isinstance(x, (dict, list, tuple)):
        return (
            {k: clean(v) for k, v in x.items()}
            if isinstance(x, dict)
            else [clean(v) for v in x]
        )
    if isinstance(x, np.ndarray):
        return clean(x.tolist())
    if isinstance(x, np.generic):
        return clean(x.item())
    if isinstance(x, float) and not np.isfinite(x):
        return None
    return x


def save(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".new")
    tmp.write_text(json.dumps(clean(data), allow_nan=False, indent=1) + "\n")
    tmp.replace(path)


def registered_case(geom, native_root, solver_registration, policy, settings_fields):
    header = native_root / "geometries" / f"{geom}.json"
    meta = json.loads(header.read_text())
    geometry = Path(meta["geometry_file"])
    assert (
        meta["split"] == "train"
        and meta["id"] == geom
        and sha(geometry) == meta["geometry_sha256"]
    )
    (raw,) = [Path(k) for k in meta["files"] if Path(k).suffix.lower() == ".ply"]
    assert sha(raw) == meta["files"][str(raw)]
    # Bootstrap the exact Python extension path used by the original native paired solver.
    sys.path.insert(0, str(solver_registration))
    import paired_case

    spin = paired_case.spin_module()
    from make_it_stand.verification.cherchi_inprocess import InProcessCherchiEvaluator
    import trimesh

    mesh = trimesh.load(raw, process=False, maintain_order=True)
    physical = np.asarray(mesh.vertices)
    norm = meta["target_frame"]["normalization"]
    vertices = (physical - np.asarray(norm["origin"])) / norm["L"]
    with np.load(geometry, allow_pickle=False) as z:
        faces = z["faces"]
        density = float(z["density"])
        origin = z["target_origin"]
        axes = z["target_axes"]
        fixed = z["fixed_moments"]
        assert np.array_equal(
            (vertices - origin) @ axes, z["vertices"]
        ) and np.array_equal(faces, mesh.faces)
    settings = spin.SpinSolverConfig(
        **settings_fields, deterministic_seed=policy["seed"]
    )
    req = spin.SpinItFasterRequest(
        vertices,
        faces,
        vertices,
        faces,
        target_origin=origin,
        target_axes=axes,
        fixed_shell_moments=fixed,
        material_density=density,
        original_units_per_normalized_unit=norm["L"],
        settings=settings,
    )
    assert (
        req.mode == policy["registered_request_mode"]
        and req.static_objective == policy["registered_objective"]
    )
    opts = spin.SpinEvaluatorConfig(**meta["profile"])
    backend = InProcessCherchiEvaluator(
        replace(req, settings=spin.SpinSolverConfig()),
        opts,
        normalization=norm,
        design_vertices_physical=physical.tolist(),
        rational_arithmetic_backend="NATIVE_GMP_SURFACE_V1",
        fixed_boundary_reuse=True,
        integral_profile=meta["integral_profile"],
    )
    assert backend.context_identity == meta["evaluator_identity"]
    files = {
        str(a): sha(a)
        for a in [
            Path(__file__),
            PROTOCOL,
            header,
            geometry,
            raw,
            Path(spin.__file__),
            Path(inspect.getfile(InProcessCherchiEvaluator)),
            Path(backend.native.__file__),
            paired_case.JOB / "launch.json",
        ]
    }
    return req, spin, meta, backend, files, float(np.ptp(vertices, axis=0).max())


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--native-root", type=Path, default=Path("/tmp/tvcg-v21-native"))
    p.add_argument("--solver-registration", type=Path, default=Path(DEFAULT_SOLVER))
    p.add_argument("--geometry", choices=["T2", "N002", "S001"], default="T2")
    p.add_argument("--start", choices=["seed", "registered"], default="seed")
    p.add_argument("--query-id", help="Named supplemental registered T2 start")
    p.add_argument(
        "--extended",
        action="store_true",
        help="Continue the T2 registered prefix under the recorded extended cap",
    )
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    policy = json.loads(PROTOCOL.read_text())
    geom = args.geometry
    settings_fields = dict(policy["solver_prefix"])
    if args.extended:
        if (geom, args.start) != ("T2", "registered"):
            raise ValueError("Only the registered T2 prefix has an extended cap")
        settings_fields.update(
            {k: v for k, v in policy["extended_T2_registered"].items() if k != "reason"}
        )
    req, spin, meta, backend, files, length = registered_case(
        geom, args.native_root, args.solver_registration, policy, settings_fields
    )
    initial = []
    if args.start == "registered":
        row_id = "T2_ellipsoid_plus" if geom == "T2" else geom + "_registered_ellipsoid"
        if args.query_id:
            if (
                geom != "T2"
                or args.query_id not in policy["other_registered_T2_starts"]
            ):
                raise ValueError("unregistered supplemental start")
            row_id = args.query_id
        rowfile = args.native_root / "rows" / f"{row_id}.json"
        row = json.loads(rowfile.read_text())
        assert row["split"] == "train" and row["geometry_id"] == geom
        files[str(rowfile)] = sha(rowfile)
        initial = [row["query"]]
    outcome = dict(
        schema="native.training-side.newton-prefix.v1",
        geometry=geom,
        split="train",
        mode=req.mode,
        objective=req.static_objective,
        start=args.start,
        query_id=(row_id if initial else None),
        seed=policy["seed"],
        explicit_guesses=initial,
        files=files,
        boolean=meta["boolean_identity"],
        profile=meta["profile"],
        evaluator_identity=backend.context_identity,
        arithmetic=meta["arithmetic"],
        integral=meta["integral_profile"],
        settings=settings_fields,
        full_editable=meta["mesh_full_editable"],
        length=length,
        events=[],
        calls=[],
    )

    def record(q):
        t = perf_counter()
        call = dict(q=list(q), index=len(outcome["calls"]) + 1)
        try:
            result = backend(q)
            call.update(
                M=result["raw_moments"],
                J=result["moment_jacobian"],
                geometry_hash=result["geometry_hash"],
                topology_hash=result["topology_hash"],
            )
            backend._materials.clear()
            return result
        except (ValueError, RuntimeError) as e:
            call["failure"] = str(e)
            raise
        finally:
            call["seconds"] = perf_counter() - t
            outcome["calls"].append(call)
            save(args.output, outcome)

    def sink(e):
        outcome["events"].append(e)
        save(args.output, outcome)
        print(
            geom,
            args.start,
            e["type"],
            e["iteration"],
            e["reason"],
            e.get("raw_equation_residual"),
            flush=True,
        )

    try:
        res = spin.solve_spin_it_faster_with_external_moment_evaluator(
            req, record, initial_quadrics=initial, event_sink=sink
        )
        outcome["result"] = {
            k: res[k]
            for k in [
                "status",
                "diagnostic",
                "function_evaluations",
                "trace_hash",
                "event_hash",
                "restart_count",
                "trace",
            ]
        }
    except (RuntimeError, ValueError) as e:
        outcome["native_rejection"] = dict(type=type(e).__name__, message=str(e))
    save(args.output, outcome)


if __name__ == "__main__":
    main()
