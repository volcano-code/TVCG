"""Exercise Work A export -> actual training -> frozen research screening.

Every reference in this command is an analytic fixture. The development box is
physically scaled, not just renamed. This verifies transport and execution; it
does not constitute held-out scientific evidence, deployed-teacher validation,
or permission for a production training run.
"""
from __future__ import annotations

import argparse
from importlib import import_module
import json
from pathlib import Path

import numpy as np

from .baseline_cases import (
    DEGREE, LABEL_IDENTITY_FIELDS, PREFIX, operand_fingerprint, query_fingerprint, sha256,
)
from .export_cases import EXPORT_SCHEMA, QUALIFICATION_SCHEMA, RAW_IDENTITY_FIELDS, export_cases
from .promotion_gate import PROTOCOL_SCHEMA, load_and_screen
from ..training.manifest import load_training_manifest
from ..training.runner import Trainer, TrainConfig
from ..training.smoke import make_smoke_manifest


def _write(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf8")


def _scaled_case(source_path, directory, factor=.9):
    """Exact uniform scaling of analytic moments and their physical chart J."""
    spec = json.loads(Path(source_path).read_text())
    source_root = Path(source_path).parent
    directory.mkdir()
    with np.load(source_root / spec["assets"]["geometry"]["path"], allow_pickle=False) as data:
        vertices, faces = data["vertices"] * factor, data["faces"].copy()
    np.savez(directory / "geometry.npz", vertices=vertices, faces=faces)
    material_module = import_module(PREFIX + ".v21_material")
    original_material = material_module.load_material(source_root / spec["assets"]["material"]["path"])
    material = material_module.prepare_material(vertices, faces, original_material.origin * factor,
        original_material.pitch * factor, side=256, density=original_material.density)
    material.save(directory / "material.npz")
    native = import_module(PREFIX + ".v21_native")
    original = native.FiniteOperand.load(source_root / "operand.npz")
    velocity = original.vertex_velocity.copy()
    # q=(physical offset, dimensionless tilts): translating the cutting plane
    # remains one length per offset, while a tilt multiplies scaled y/z.
    velocity[:, :, 1:] *= factor
    operand = native.FiniteOperand(original.vertices * factor, original.faces,
                                   original.face_is_cap, original.operation, velocity)
    operand.save(directory / "operand.npz")
    with np.load(source_root / "targets.npz", allow_pickle=False) as data:
        targets = {name: data[name].copy() for name in data.files}
    for name in ("fixed_moments", "native_moments", "native_full_editable_moments"):
        targets[name] *= factor ** (3 + DEGREE)
    targets["native_jacobian"] *= (factor ** (3 + DEGREE))[:, None]
    targets["native_jacobian"][:, 0] /= factor
    targets["coefficients"][0] *= factor
    targets["parameter_scales"][0] *= factor
    targets["direction"][0] *= factor
    meta = spec["meta"]
    meta.update(geometry_id="synthetic_scaled_development_box", source_group="analytic_scaled_fixture",
                query_id="scaled_finite_plane", geometry_sha256=sha256(directory / "geometry.npz"),
                material_identity=material.identity, operand_content_sha256=operand_fingerprint(operand),
                query_content_sha256=query_fingerprint(targets["coefficients"]),
                editable_mass=float(targets["native_full_editable_moments"][0]),
                length_scale=meta["length_scale"] * factor)
    targets["metadata"] = np.asarray(json.dumps({key: meta[key] for key in LABEL_IDENTITY_FIELDS}))
    np.savez(directory / "targets.npz", **targets)
    spec["assets"] = {name: dict(path=name + ".npz", sha256=sha256(directory / (name + ".npz")))
                      for name in ("geometry", "material", "operand", "targets")}
    path = directory / "case.json"; _write(path, spec)
    return path


def prepare_handoff(directory):
    directory = Path(directory)
    original_manifest = make_smoke_manifest(directory)
    entries = json.loads(original_manifest.read_text())["records"]
    scaled = _scaled_case(directory / entries[0]["case_path"], directory / "development")
    entries.append(dict(record_id="fixture-development", case_path=str(scaled.relative_to(directory)), role="development"))
    evidence = directory / "analytic_reference.json"
    _write(evidence, dict(scope="synthetic_software_only", reference="closed-form cuboid moments and moving planar section integrals",
                         scaled_reference="uniform-coordinate scaling with physical offset and dimensionless tilt chart"))
    records = []
    material_module = import_module(PREFIX + ".v21_material")
    for item in entries:
        case_path = directory / item["case_path"]
        spec = json.loads(case_path.read_text())
        meta = {key: value for key, value in spec["meta"].items() if key != "material_identity"}
        assets = {}
        for name, source in spec["assets"].items():
            path = Path(source["path"])
            path = path if path.is_absolute() else case_path.parent / path
            assets[name] = dict(path=str(path.resolve()), sha256=sha256(path))
        material = material_module.load_material(assets["material"]["path"])
        qualification = case_path.parent / "export_qualification.json"
        _write(qualification, dict(schema=QUALIFICATION_SCHEMA, record_id=item["record_id"],
            identity={key: meta[key] for key in RAW_IDENTITY_FIELDS},
            assets_sha256={key: assets[key]["sha256"] for key in ("geometry", "operand", "targets")},
            moment_valid=True, restricted_event_valid=1, ambient_event_valid=-1,
            neighbourhood=dict(scope="fixed_rank_chart", radius=1e-4,
                               radius_definition="analytic physical-plane offset/tilts L2 ball away from material face events"),
            evidence=[dict(path=str(evidence.resolve()), sha256=sha256(evidence))]))
        assets["qualification"] = dict(path=str(qualification.resolve()), sha256=sha256(qualification))
        records.append(dict(record_id=item["record_id"], role=item["role"], meta=meta, assets=assets,
            grid=dict(origin=material.origin.tolist(), pitch=float(material.pitch), side=256),
            surface_family="plane", trajectory_stage="synthetic_query", moment_weight=1., jacobian_weight=1.))
    request = directory / "export_request.json"
    _write(request, dict(schema=EXPORT_SCHEMA, snapshot_id="synthetic-export-training-screen-v1",
                         registered_train_count=2, records=records))
    return request


def run(output):
    output = Path(output).resolve(); output.mkdir(parents=True, exist_ok=False)
    request = prepare_handoff(output / "input")
    manifest = export_cases(request, output / "exported", synthetic=True)
    # Register exact evaluation membership/aggregation BEFORE predictions.
    registered = Path(__file__).resolve().parents[1] / "protocol.json"
    protocol = output / "screen_protocol.json"
    _write(protocol, dict(schema=PROTOCOL_SCHEMA, protocol_id="synthetic-software-pipeline-only",
        registered_protocol_sha256=sha256(registered), primary_aggregation="source_macro_group_rmse",
        jacobian_aggregation="source_macro_group_rmse", primary_relative_improvement=.05,
        protection_max_ratio=1.05, protected_retention_metric="scaled10_moment_rmse",
        protected_jacobian_metric="scaled_jacobian_rmse", require_no_learning_baseline=True,
        evaluation_steps=[1, 2],
        expected_record_ids=["fixture-development"]))
    data = load_training_manifest(manifest, mode="smoke", cache_directory=output / "cache")
    trainer = Trainer(data, output / "training", config=TrainConfig(), mode="smoke", device="cpu",
                      max_steps=2, max_attempts=4, run_id="synthetic-export-training-screen",
                      evaluation_protocol_path=protocol, registered_protocol_path=registered)
    training = trainer.run(checkpoint_every=1, evaluate_every=1)
    screening = load_and_screen(training["screen"]["bundle_path"])
    checks = dict(real_updates=training["global_step"] == 2,
                  distinct_geometry_hashes=data.train[0]["case_meta"]["geometry_sha256"] != data.development[0]["case_meta"]["geometry_sha256"],
                  candidate_is_post_update=json.loads((output / "training" / "best.json").read_text())["best_global_step"] > 0,
                  original_teacher_retained="source_teacher_targets" in data.train[0]["case"]["asset_receipts"],
                  matched_evaluation_membership=screening["coverage"]["registered_queries"] == 1,
                  missing_full_j_is_explicit="Full-J" in screening["insufficient_evidence"],
                  synthetic_never_authorizes_long=screening["training_authorized"] is False and screening["eligible_for_long_training_review"] is False,
                  no_physics_or_speed_claim=screening["physical_acceptance"] == "NOT_EVALUATED" and screening["speedup_claim"] is False)
    report = dict(schema="tvcg.v22.export-training-screen-smoke.v1", scope="synthetic_software_only",
                  status="PASS" if all(checks.values()) else "FAIL", checks=checks,
                  manifest=str(manifest), manifest_sha256=sha256(manifest),
                  training_result=str(output / "training" / "result.json"),
                  screen_report=training["screen"]["report_path"], screen_decision=screening["decision"],
                  geometry_evaluation="two training queries and one scaled analytic development geometry; no generalization claim",
                  real_training_dataset_used=False, native_mesh_acceptance_evaluated=False)
    _write(output / "integration_smoke_report.json", report)
    if not all(checks.values()):
        raise AssertionError(checks)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    print(json.dumps(run(args.output), indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
