"""Export a frozen Work A teacher handoff to executable conservative cases.

This importer does not manufacture teacher labels or derivative qualifications.
The original mesh NPZ, finite operand NPZ, target NPZ, and qualification receipt
are independently content addressed.  Only the static material representation
is computed here.  A qualified numerical Jacobian needs its neighbourhood
receipt; a finite matrix alone cannot promote a row into Jacobian supervision.

See INTEGRATION.md for the deliberately explicit handoff schemas.  Destinations
are new directories; a complete manifest is published only after every admitted
case has passed identity and full-material checks.  Diagnostic/locked sources
never enter the training manifest.  Synthetic imports require a separate flag.
"""
from __future__ import annotations

import argparse
from collections import Counter
from importlib import import_module
import json
from pathlib import Path
import shutil
import time

import numpy as np

from .baseline_cases import (
    COEFFICIENT_MULTIPLIERS, LABEL_IDENTITY_FIELDS, ORDER, PREFIX, SCHEMA,
    _checked_fixed, finite_array, jsonable, moment_scale, operand_fingerprint,
    query_fingerprint, sha256, validate_meta, verify_file_receipt,
)

EXPORT_SCHEMA = "tvcg.v22.case-export.v1"
QUALIFICATION_SCHEMA = "tvcg.v22.teacher-qualification.v1"
TRAINING_SCHEMA = "compact_pino.training-manifest.v1"
RAW_IDENTITY_FIELDS = tuple(k for k in LABEL_IDENTITY_FIELDS if k != "material_identity")


def _read(path):
    return json.loads(Path(path).read_text(encoding="utf8"))


def _write(path, value):
    with Path(path).open("x", encoding="utf8") as stream:
        json.dump(jsonable(value), stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")


def _asset(entry, root, name):
    path = Path(entry["path"])
    path = (path if path.is_absolute() else root / path).resolve(strict=True)
    digest = sha256(path)
    if digest != entry.get("sha256"):
        raise ValueError(f"{name} SHA256 mismatch")
    return path


def _copy_asset(path, directory, suffix):
    digest = sha256(path)
    destination = directory / (digest + suffix)
    if destination.exists():
        if sha256(destination) != digest:
            raise ValueError("content-addressed output asset was modified")
    else:
        shutil.copyfile(path, destination)
    return dict(path="../assets/" + destination.name, sha256=digest)


def _same_identity(reference, meta, fields, name):
    for key in fields:
        if key not in reference or reference[key] != meta[key]:
            raise ValueError(f"{name} identity mismatch: {key}")


def _qualification_fields(q, asset_digests, meta, record_id):
    if q.get("schema") != QUALIFICATION_SCHEMA or q.get("record_id") != record_id:
        raise ValueError("qualification receipt schema or record_id mismatch")
    _same_identity(q.get("identity", {}), meta, RAW_IDENTITY_FIELDS, "qualification")
    for key in ("geometry", "operand", "targets"):
        if q.get("assets_sha256", {}).get(key) != asset_digests[key]:
            raise ValueError(f"qualification {key} binding mismatch")
    if q.get("moment_valid") is not True:
        raise ValueError("teacher has not admitted this Moment row")
    evidence = q.get("evidence", [])
    if not isinstance(evidence, list) or not evidence:
        raise ValueError("qualification needs at least one original evidence artifact")
    # One-point numerical validity does not imply an open valid neighbourhood.
    if meta["derivative_valid"]:
        key = "ambient_event_valid" if meta["derivative_dimension"] == 10 else "restricted_event_valid"
        if type(q.get(key)) is not int or q[key] != 1:
            raise ValueError(f"qualified derivative requires {key}=1")
        neighbourhood = q.get("neighbourhood", {})
        expected = "ambient_raw_coefficients" if meta["derivative_dimension"] == 10 else "fixed_rank_chart"
        radius = neighbourhood.get("radius", 0)
        if (neighbourhood.get("scope") != expected or isinstance(radius, bool)
                or not isinstance(radius, (float, int)) or not np.isfinite(radius) or radius <= 0
                or not isinstance(neighbourhood.get("radius_definition"), str)
                or not neighbourhood["radius_definition"].strip()):
            raise ValueError("qualified derivative needs an explicit positive valid-neighbourhood radius")


def _qualification(path, assets, meta, record_id, root):
    q = _read(path)
    _qualification_fields(q, {key: sha256(assets[key]) for key in ("geometry", "operand", "targets")}, meta, record_id)
    evidence_paths = [_asset(item, root, "qualification evidence") for item in q["evidence"]]
    return q, evidence_paths


def verify_export_qualification(case, record_id, *, asset_verification_cache=None):
    """Verify a portable exported declaration, without independently certifying J.

    Return None when no receipt was supplied. A supplied but contradictory or
    changed receipt is an error, never silently downgraded to a valid label.
    Original evidence paths can belong to the Work host; their verified copies
    are resolved by SHA through this case's supporting-asset receipts.
    """
    receipts = case.get("asset_receipts", {})
    if "source_qualification" not in receipts:
        return None
    def checked(name):
        if name not in receipts:
            raise ValueError("qualification supporting asset missing: " + name)
        receipt = receipts[name]
        verified = verify_file_receipt(receipt["path"], receipt.get("sha256"),
                                       cache=asset_verification_cache, name="qualification " + name)
        return Path(verified["path"])
    qpath = checked("source_qualification")
    q = _read(qpath)
    assets = dict(geometry=checked("geometry"), operand=checked("operand"), targets=checked("source_teacher_targets"))
    _qualification_fields(q, {"geometry": receipts["geometry"]["sha256"], "operand": receipts["operand"]["sha256"],
                             "targets": receipts["source_teacher_targets"]["sha256"]}, case["meta"], record_id)
    declared = {item.get("sha256") for item in q["evidence"]}
    available = set()
    for key in receipts:
        if key.startswith("source_evidence_"):
            checked(key)
            available.add(receipts[key]["sha256"])
    if declared != available:
        raise ValueError("qualification evidence copies differ from the original receipt")
    # Newly packed targets must contain the exact original teacher values. A
    # matching metadata string by itself must not bless altered numeric labels.
    names = ["coefficients", "fixed_moments", "native_moments", "native_full_editable_moments"]
    if case["meta"]["derivative_valid"]:
        names += ["native_jacobian", "chart_basis", "parameter_scales"]
        if "direction" in case:
            names.append("direction")
    with np.load(assets["targets"], allow_pickle=False) as original:
        _same_identity(json.loads(str(original["metadata"].item())), case["meta"], RAW_IDENTITY_FIELDS, "original teacher")
        for name in names:
            if name not in original or not np.array_equal(np.asarray(case[name]), original[name]):
                raise ValueError("exported numeric target differs from original teacher: " + name)
    return dict(status="BOUND_EXTERNAL_DECLARATION_VERIFIED", path=str(qpath), sha256=receipts["source_qualification"]["sha256"],
                declared_derivative_valid=case["meta"]["derivative_valid"],
                independent_teacher_or_neighbourhood_reaudit=False)


def _row_meta(record, synthetic):
    meta = dict(record["meta"])
    if record.get("role") not in ("train", "development"):
        raise ValueError("only train/development roles can be exported for this runner")
    if synthetic:
        if meta.get("split") != "synthetic" or meta.get("reference_kind") != "analytic_fixture":
            raise ValueError("synthetic export requires explicitly synthetic fixture identities")
    elif meta.get("split") != record["role"] or meta.get("reference_kind") != "deployed_native":
        raise ValueError("real export requires a matching train/development deployed-native identity")
    if not isinstance(record.get("record_id"), str) or not record["record_id"].strip():
        raise ValueError("stable nonempty record_id required")
    if tuple(meta.get("moment_order", ())) != ORDER or tuple(meta.get("coefficient_multipliers", ())) != COEFFICIENT_MULTIPLIERS:
        raise ValueError("moment/coefficient ordering mismatch")
    # A placeholder only allows validation of the raw metadata before building
    # material. It is never written to any exported record.
    validate_meta(dict(meta, material_identity="0" * 64))
    if meta["length_scale_definition"] != "editable_bbox_max_extent":
        raise ValueError("this export contract uses editable_bbox_max_extent")
    for name in ("moment_weight", "jacobian_weight"):
        value = record.get(name, 1.)
        if isinstance(value, bool) or not np.isfinite(value) or value < 0:
            raise ValueError(f"{name} must be finite and nonnegative")
    if record.get("loader_eligible", True) is not True:
        raise ValueError("export only admitted rows; retain rejected rows in the separate upstream receipt")
    return meta


def _build_case(record, meta, root, output, materials):
    assets = {key: _asset(record["assets"][key], root, key)
              for key in ("geometry", "operand", "targets", "qualification")}
    if sha256(assets["geometry"]) != meta["geometry_sha256"]:
        raise ValueError("geometry identity mismatch")
    qualification, evidence = _qualification(assets["qualification"], assets, meta, record["record_id"], root)
    native = import_module(PREFIX + ".v21_native")
    material_module = import_module(PREFIX + ".v21_material")
    operand = native.FiniteOperand.load(assets["operand"])
    if operand_fingerprint(operand) != meta["operand_content_sha256"] or operand.operation != meta["operation"]:
        raise ValueError("finite operand identity or operation mismatch")
    with np.load(assets["geometry"], allow_pickle=False) as data:
        vertices, faces = data["vertices"].copy(), data["faces"].copy()
    # The source geometry must already be in the registered solver frame. This
    # importer never guesses units, recentres, or rescales the original mesh.
    vertices, faces = material_module._mesh(vertices, faces)
    length = float(np.ptp(vertices, axis=0).max())
    if not np.isclose(length, meta["length_scale"], rtol=1e-12, atol=0):
        raise ValueError("mesh extent differs from registered physical length scale")
    with np.load(assets["targets"], allow_pickle=False) as data:
        raw_meta = json.loads(str(data["metadata"].item()))
        _same_identity(raw_meta, meta, RAW_IDENTITY_FIELDS, "teacher target")
        arrays = {key: data[key].copy() for key in ("coefficients", "fixed_moments", "native_moments", "native_full_editable_moments")}
        if meta["derivative_valid"]:
            arrays.update({key: data[key].copy() for key in ("native_jacobian", "chart_basis", "parameter_scales")})
            if "direction" in data:
                arrays["direction"] = data["direction"].copy()
    for key in ("coefficients", "fixed_moments", "native_moments", "native_full_editable_moments"):
        finite_array(arrays[key], (10,), key)
    if query_fingerprint(arrays["coefficients"]) != meta["query_content_sha256"]:
        raise ValueError("raw coefficient identity mismatch")
    scale = moment_scale(meta["editable_mass"], meta["length_scale"])
    _checked_fixed(arrays["fixed_moments"], scale)
    full = arrays["native_full_editable_moments"]
    if not np.isclose(full[0], meta["editable_mass"], rtol=1e-12, atol=0):
        raise ValueError("normalization mass differs from full editable teacher mass")
    exact = material_module.signed_mesh_moments(vertices, faces, meta["density"])
    mesh_error = float(np.max(np.abs(exact - full) / scale))
    if exact[0] <= 0 or mesh_error > 1e-8:
        raise ValueError(f"teacher/independent-mesh full ten-moment mismatch: {mesh_error}")
    # Check the material retained mass, not the total mass including fixed shell.
    retained_mass = arrays["native_moments"][0] - arrays["fixed_moments"][0]
    if retained_mass < -1e-8 * full[0] or retained_mass > full[0] * (1 + 1e-8):
        raise ValueError("editable retained mass is outside its physical interval")
    if meta["derivative_valid"]:
        dim = meta["derivative_dimension"]
        finite_array(operand.vertex_velocity, (len(operand.vertices), 3, dim), "vertex_velocity")
        finite_array(arrays["native_jacobian"], (10, dim), "native_jacobian")
        chart = finite_array(arrays["chart_basis"], (10, dim), "chart_basis")
        if np.linalg.matrix_rank(chart) != dim or (dim == 10 and not np.array_equal(chart, np.eye(10))):
            raise ValueError("invalid typed chart; Full-J requires the raw identity chart")
        if np.any(finite_array(arrays["parameter_scales"], (dim,), "parameter_scales") <= 0):
            raise ValueError("positive chart parameter scales required")
        if "direction" in arrays:
            finite_array(arrays["direction"], (dim,), "direction")
            allowed = "synthetic_fixture" if meta["split"] == "synthetic" else "native_newton"
            if meta.get("direction_source") != allowed:
                raise ValueError("an actual direction requires its registered native/synthetic source")
    grid = record["grid"]
    if type(grid.get("side")) is not int or grid["side"] != 256:
        raise ValueError("the current export supports native256 only")
    origin, pitch, side, density = material_module._grid(grid["origin"], grid["pitch"], 256, meta["density"])
    material_id = material_module._identity("mesh", origin, pitch, side, density, vertices, faces)
    if material_id not in materials:
        if "material" in record["assets"]:
            # Static preprocessing can be reused across snapshots, but only
            # for the exact mesh, frame/grid, density and implementation revision.
            previous_material = _asset(record["assets"]["material"], root, "material")
            material = material_module.load_material(previous_material, material_id)
        else:
            material = material_module.prepare_material(vertices, faces, origin, pitch, side=side, density=density)
        material_file = output / "assets" / (material_id + ".material.npz")
        material.save(material_file)
        # Keep only the small static receipt between records. Holding every
        # geometry's boundary quadrature in RAM would scale with the full pool.
        materials[material_id] = (material.moments(), material_file)
    material_moments, material_file = materials[material_id]
    material_error = float(np.max(np.abs(material_moments - full) / scale))
    if material_error > 1e-8:
        raise ValueError(f"conservative cache/teacher full-moment mismatch: {material_error}")
    meta["material_identity"] = material_id
    validate_meta(meta)
    # The raw teacher archive remains unmodified; final targets bind the newly
    # computed static material identity in addition to all original identities.
    arrays["metadata"] = np.asarray(json.dumps({key: meta[key] for key in LABEL_IDENTITY_FIELDS}))
    index = record["record_id"]
    import hashlib
    identifier = hashlib.sha256(index.encode("utf8")).hexdigest()[:24]
    target_file = output / "assets" / (identifier + ".targets.npz")
    np.savez_compressed(target_file, **arrays)
    case_assets = {key: _copy_asset(assets[key], output / "assets", "." + key + ".npz") for key in ("geometry", "operand")}
    case_assets["material"] = dict(path="../assets/" + material_file.name, sha256=sha256(material_file))
    case_assets["targets"] = dict(path="../assets/" + target_file.name, sha256=sha256(target_file))
    qualification_copy = _copy_asset(assets["qualification"], output / "assets", ".qualification.json")
    evidence_copies = [_copy_asset(p, output / "assets", ".evidence" + p.suffix) for p in evidence]
    raw_teacher_copy = _copy_asset(assets["targets"], output / "assets", ".teacher.npz")
    block_width = record.get("block_voxels", 8)
    if type(block_width) is not int or block_width <= 0 or 256 % block_width:
        raise ValueError("block_voxels must divide 256")
    case_file = output / "cases" / (identifier + ".json")
    _write(case_file, dict(schema=SCHEMA, meta=meta, block_voxels=block_width, assets=case_assets,
                         source_qualification=qualification_copy, source_evidence=evidence_copies,
                         source_teacher_targets=raw_teacher_copy))
    manifest_row = dict(record_id=index, case_path="cases/" + case_file.name, case_sha256=sha256(case_file),
                        role=record["role"], loader_eligible=True,
                        moment_weight=float(record.get("moment_weight", 1.)),
                        jacobian_weight=float(record.get("jacobian_weight", 1.)))
    for key in ("surface_family", "trajectory_stage"):
        value = record.get(key)
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"explicit {key} is required (use unknown when evidence is missing)")
        manifest_row[key] = value
    ratio = float(retained_mass / full[0])
    bin_name = "near_zero" if ratio < .01 else "low" if ratio < .10 else "middle" if ratio < .70 else "high"
    if record.get("retention_bin", bin_name) != bin_name:
        raise ValueError("registered retention_bin disagrees with editable teacher mass")
    manifest_row["retention_bin"] = bin_name
    receipt = dict(record_id=index, geometry_id=meta["geometry_id"], query_id=meta["query_id"],
                   source_group=meta["source_group"], role=record["role"], **{k: manifest_row[k] for k in
                   ("surface_family", "trajectory_stage", "retention_bin")},
                   retained_fraction=ratio, derivative_valid=meta["derivative_valid"],
                   derivative_dimension=meta["derivative_dimension"], chart_id=meta["chart_id"],
                   mesh_teacher_full_scaled_max=mesh_error, cache_teacher_full_scaled_max=material_error,
                   teacher_qualification_sha256=qualification_copy["sha256"],
                   qualification_is_external_assertion=True)
    return manifest_row, receipt


def export_cases(request_path, output, *, synthetic=False):
    started = time.perf_counter()
    request_path = Path(request_path).resolve(strict=True)
    request, output = _read(request_path), Path(output).resolve()
    if request.get("schema") != EXPORT_SCHEMA:
        raise ValueError("unsupported export request schema")
    if not isinstance(request.get("snapshot_id"), str) or not request["snapshot_id"].strip():
        raise ValueError("snapshot_id required")
    rows = request.get("records")
    if not isinstance(rows, list) or not rows:
        raise ValueError("nonempty admitted records required")
    metas = [_row_meta(row, synthetic) for row in rows]  # Refuse locked sources before reading targets.
    from .handoff_authority import verify_handoff_authority
    authority = verify_handoff_authority(request, request_path)
    if (authority["receipt"]["status"] == "NOT_APPLICABLE_SYNTHETIC") != synthetic:
        raise ValueError("real/synthetic authority scope differs from the requested import")
    ids = [r["record_id"] for r in rows]
    keys = [(m["geometry_id"], m["query_id"], m["chart_id"]) for m in metas]
    if len(set(ids)) != len(ids) or len(set(keys)) != len(keys):
        raise ValueError("duplicate record or query/chart identity")
    groups = {role: {m["source_group"] for r, m in zip(rows, metas) if r["role"] == role}
              for role in ("train", "development")}
    geometries = {role: {m["geometry_id"] for r, m in zip(rows, metas) if r["role"] == role}
                  for role in ("train", "development")}
    geometry_hashes = {role: {m["geometry_sha256"] for r, m in zip(rows, metas) if r["role"] == role}
                       for role in ("train", "development")}
    if (groups["train"] & groups["development"] or geometries["train"] & geometries["development"]
            or geometry_hashes["train"] & geometry_hashes["development"]):
        raise ValueError("training and development must have disjoint sources and geometries")
    registered = request.get("registered_train_count")
    if type(registered) is not int or registered < sum(r["role"] == "train" for r in rows):
        raise ValueError("registered_train_count must cover admitted training records")
    output.mkdir(parents=True, exist_ok=False)
    (output / "assets").mkdir(); (output / "cases").mkdir()
    material_cache, manifest_rows, receipts = {}, [], []
    try:
        packaged_authority = None
        if not synthetic:
            def portable_asset(receipt, suffix):
                result = _copy_asset(Path(receipt["path"]), output / "assets", suffix)
                result["path"] = "assets/" + Path(result["path"]).name
                return result
            authority_receipt = authority["receipt"]
            packaged_authority = portable_asset(authority_receipt["authority"], ".authority.json")
            packaged_authority["supporting_assets"] = {
                role: portable_asset(receipt, ".authority_" + role + Path(receipt["path"]).suffix)
                for role, receipt in authority_receipt["verified_assets"].items()}
            packaged_authority["teacher_contract_sha256"] = authority_receipt["teacher_contract_sha256"]
        for row, meta in zip(rows, metas):
            exported, receipt = _build_case(row, meta, request_path.parent, output, material_cache)
            manifest_rows.append(exported); receipts.append(receipt)
        counts = Counter((r["role"], r["source_group"], r["surface_family"], r["retention_bin"], r["trajectory_stage"],
                          ("R" + str(r["derivative_dimension"]) if r["derivative_valid"] else "Moment_only")) for r in receipts)
        coverage = [dict(zip(("role", "source_group", "surface_family", "retention_bin", "trajectory_stage", "derivative"), key), count=count)
                    for key, count in sorted(counts.items())]
        receipt = dict(schema="tvcg.v22.case-export-receipt.v1", status="EXPORTED",
                       scope="synthetic_software_only" if synthetic else "frozen_teacher_import_not_independent_teacher_certification",
                       input_request_sha256=sha256(request_path), snapshot_id=request["snapshot_id"],
                       registered_train_count=registered, exported_count=len(rows), static_material_count=len(material_cache),
                       seconds=time.perf_counter() - started, coverage=coverage, records=receipts,
                       source_admission=authority["receipt"])
        _write(output / "export_receipt.json", receipt)
        manifest = dict(schema=TRAINING_SCHEMA, snapshot_id=request["snapshot_id"],
               registered_train_count=registered, records=manifest_rows,
               export_receipt_sha256=sha256(output / "export_receipt.json"))
        if packaged_authority is not None:
            manifest["source_authority"] = packaged_authority
        _write(output / "training_manifest.json", manifest)
        return output / "training_manifest.json"
    except BaseException as exc:
        # Preserve the partial evidence for debugging, but there is no runnable
        # training manifest after any validation/build failure.
        _write(output / "EXPORT_FAILED.json", dict(status="STOP", error_type=type(exc).__name__, error=str(exc),
                 completed_records=[r["record_id"] for r in receipts], input_request_sha256=sha256(request_path)))
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--request", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--synthetic-fixture", action="store_true")
    args = parser.parse_args()
    print(export_cases(args.request, args.output, synthetic=args.synthetic_fixture))


if __name__ == "__main__":
    main()
