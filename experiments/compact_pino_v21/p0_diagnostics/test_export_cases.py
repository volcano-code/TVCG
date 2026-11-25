"""Physical handoff tests using the actual conservative native256 builder.

Only synthetic references are constructed here. They do not assert Work A's
real data admission, a deployed Boolean derivative, or a production checkpoint.
"""
import copy
import importlib
import json
from pathlib import Path
import shutil

import numpy as np
import pytest

from .baseline_cases import load_baseline_case, sha256, write_synthetic_finite_case
from .export_cases import EXPORT_SCHEMA, QUALIFICATION_SCHEMA, RAW_IDENTITY_FIELDS, export_cases, verify_export_qualification


@pytest.fixture(scope="module")
def source(tmp_path_factory):
    folder = tmp_path_factory.mktemp("work_a_fixture") / "raw"
    case_path = write_synthetic_finite_case(folder)
    return case_path


def request(tmp_path, source, *, qualified=True):
    case = json.loads(source.read_text())
    meta = {k: v for k, v in case["meta"].items() if k != "material_identity"}
    meta["derivative_valid"] = qualified
    if not qualified:
        meta["derivative_reason"] = "synthetic_test_withholds_event_qualification"
    assets = {k: {"path": str(source.parent / case["assets"][k]["path"]),
                  "sha256": case["assets"][k]["sha256"]} for k in ("geometry", "operand")}
    with np.load(source.parent / "targets.npz", allow_pickle=False) as data:
        targets = {k: data[k].copy() for k in data.files}
    targets["metadata"] = np.asarray(json.dumps({k: meta[k] for k in RAW_IDENTITY_FIELDS}))
    if not qualified:
        targets["native_jacobian"][:] = np.nan
    target_path = tmp_path / "teacher.npz"
    np.savez(target_path, **targets)
    assets["targets"] = dict(path=str(target_path), sha256=sha256(target_path))
    evidence_path = tmp_path / "analytic_receipt.txt"
    evidence_path.write_text("Synthetic analytic cuboid/plane reference; not deployed physics evidence.\n")
    q = dict(schema=QUALIFICATION_SCHEMA, record_id="box/plane/r3",
             identity={k: meta[k] for k in RAW_IDENTITY_FIELDS},
             assets_sha256={k: v["sha256"] for k, v in assets.items()}, moment_valid=True,
             restricted_event_valid=1 if qualified else -1, ambient_event_valid=-1,
             neighbourhood=dict(scope="fixed_rank_chart", radius=1e-4, radius_definition="synthetic raw-plane chart L2 radius"),
             evidence=[dict(path=str(evidence_path), sha256=sha256(evidence_path))])
    qp = tmp_path / "qualification.json"; qp.write_text(json.dumps(q))
    assets["qualification"] = dict(path=str(qp), sha256=sha256(qp))
    r = dict(record_id="box/plane/r3", role="train", meta=meta, assets=assets,
             grid=dict(origin=[-8.] * 3, pitch=1/16, side=256),
             surface_family="plane", trajectory_stage="analytic_query")
    payload = dict(schema=EXPORT_SCHEMA, snapshot_id="synthetic-software-only", registered_train_count=2, records=[r])
    rp = tmp_path / "request.json"; rp.write_text(json.dumps(payload))
    return rp, payload


def rewrite(path, obj):
    path.write_text(json.dumps(obj))


def test_import_builds_material_and_loadable_bound_case(tmp_path, source):
    rp, _ = request(tmp_path, source)
    manifest_path = export_cases(rp, tmp_path / "out", synthetic=True)
    manifest = json.loads(manifest_path.read_text())
    row = manifest["records"][0]
    case = load_baseline_case(manifest_path.parent / row["case_path"])
    assert row["case_sha256"] == sha256(manifest_path.parent / row["case_path"])
    assert case["meta"]["derivative_valid"] and case["native_jacobian"].shape == (10, 3)
    assert row["retention_bin"] == "middle" and case["fixed_moments"][0] > 0
    receipt = json.loads((manifest_path.parent / "export_receipt.json").read_text())
    assert receipt["scope"] == "synthetic_software_only"
    assert receipt["records"][0]["cache_teacher_full_scaled_max"] < 1e-8
    assert manifest["registered_train_count"] == 2  # Not invented exposure.
    assert "source_qualification" in case["asset_receipts"]
    bound = verify_export_qualification(case, row["record_id"])
    assert bound["status"] == "BOUND_EXTERNAL_DECLARATION_VERIFIED"
    assert bound["independent_teacher_or_neighbourhood_reaudit"] is False
    q = Path(case["asset_receipts"]["source_qualification"]["path"])
    q.write_text(q.read_text() + " ")
    with pytest.raises(ValueError, match="source_qualification asset SHA256"):
        load_baseline_case(manifest_path.parent / row["case_path"])


def test_unqualified_derivative_is_never_imported_as_zeros(tmp_path, source):
    rp, _ = request(tmp_path, source, qualified=False)
    manifest_path = export_cases(rp, tmp_path / "out", synthetic=True)
    manifest = json.loads(manifest_path.read_text())
    case = load_baseline_case(manifest_path.parent / manifest["records"][0]["case_path"])
    assert not case["meta"]["derivative_valid"]
    assert "native_jacobian" not in case
    target_path = case["asset_receipts"]["targets"]["path"]
    with np.load(target_path, allow_pickle=False) as data:
        assert "native_jacobian" not in data


def test_synthetic_cannot_enter_real_handoff(tmp_path, source):
    rp, _ = request(tmp_path, source)
    with pytest.raises(ValueError, match="deployed-native"):
        export_cases(rp, tmp_path / "out")
    assert not (tmp_path / "out").exists()


@pytest.mark.parametrize("mutation,expected", [
    ("locked_role", "only train/development"),
    ("geometry_hash", "SHA256"),
    ("no_event", "restricted_event_valid"),
    ("no_radius", "valid-neighbourhood radius"),
    ("wrong_density", "identity mismatch"),
    ("full_material", "independent-mesh"),
])
def test_failures_never_publish_a_training_manifest(tmp_path, source, mutation, expected):
    rp, payload = request(tmp_path, source)
    row = payload["records"][0]
    qp = Path(row["assets"]["qualification"]["path"])
    q = json.loads(qp.read_text())
    if mutation == "locked_role":
        row["role"] = "benchmark"
    elif mutation == "geometry_hash":
        row["assets"]["geometry"]["sha256"] = "a" * 64
    elif mutation == "wrong_density":
        row["meta"]["density"] = 2.
    elif mutation in ("no_event", "no_radius"):
        if mutation == "no_event":
            q["restricted_event_valid"] = -1
        else:
            q["neighbourhood"]["radius"] = 0.
        rewrite(qp, q); row["assets"]["qualification"]["sha256"] = sha256(qp)
    elif mutation == "full_material":
        target_path = Path(row["assets"]["targets"]["path"])
        with np.load(target_path, allow_pickle=False) as data:
            targets = {k: data[k].copy() for k in data.files}
        targets["native_full_editable_moments"][4] += .01
        np.savez(target_path, **targets)
        row["assets"]["targets"]["sha256"] = sha256(target_path)
        q["assets_sha256"]["targets"] = sha256(target_path)
        rewrite(qp, q); row["assets"]["qualification"]["sha256"] = sha256(qp)
    rewrite(rp, payload)
    with pytest.raises(ValueError, match=expected):
        export_cases(rp, tmp_path / "out", synthetic=True)
    assert not (tmp_path / "out" / "training_manifest.json").exists()


def test_source_leakage_stops_before_reading_labels(tmp_path, source):
    rp, payload = request(tmp_path, source)
    second = copy.deepcopy(payload["records"][0])
    second["record_id"] = "other"
    second["meta"]["query_id"] = "other"
    second["role"] = "development"
    second["assets"]["targets"]["path"] = "/does/not/exist"
    payload["records"].append(second); rewrite(rp, payload)
    with pytest.raises(ValueError, match="disjoint sources"):
        export_cases(rp, tmp_path / "out", synthetic=True)
    assert not (tmp_path / "out").exists()


def test_geometry_alias_cannot_bypass_split_isolation(tmp_path, source):
    rp, payload = request(tmp_path, source)
    second = copy.deepcopy(payload["records"][0])
    second["record_id"] = "aliased_geometry"
    second["meta"].update(geometry_id="different_id", source_group="different_name", query_id="other")
    second["role"] = "development"
    payload["records"].append(second); rewrite(rp, payload)
    with pytest.raises(ValueError, match="disjoint sources"):
        export_cases(rp, tmp_path / "out", synthetic=True)


def test_static_cache_is_reused_only_with_exact_input_identity(tmp_path, source):
    rp, payload = request(tmp_path, source)
    material_path = source.parent / "material.npz"
    payload["records"][0]["assets"]["material"] = dict(path=str(material_path), sha256=sha256(material_path))
    rewrite(rp, payload)
    manifest_path = export_cases(rp, tmp_path / "reused", synthetic=True)
    manifest = json.loads(manifest_path.read_text())
    case = load_baseline_case(manifest_path.parent / manifest["records"][0]["case_path"])
    assert case["meta"]["material_identity"] == json.loads(source.read_text())["meta"]["material_identity"]
    payload["records"][0]["grid"]["pitch"] = 1/32
    rewrite(rp, payload)
    with pytest.raises(ValueError, match="identity"):
        export_cases(rp, tmp_path / "stale", synthetic=True)


def test_packed_labels_cannot_diverge_from_the_bound_original_teacher(tmp_path, source):
    rp, _ = request(tmp_path, source)
    manifest_path = export_cases(rp, tmp_path / "out", synthetic=True)
    row = json.loads(manifest_path.read_text())["records"][0]
    case = load_baseline_case(manifest_path.parent / row["case_path"])
    case["native_moments"][3] += 1e-4
    with pytest.raises(ValueError, match="numeric target differs"):
        verify_export_qualification(case, row["record_id"])


def test_missing_evidence_copy_does_not_qualify_a_declared_derivative(tmp_path, source):
    rp, _ = request(tmp_path, source)
    manifest_path = export_cases(rp, tmp_path / "out", synthetic=True)
    row = json.loads(manifest_path.read_text())["records"][0]
    case = load_baseline_case(manifest_path.parent / row["case_path"])
    case["asset_receipts"].pop("source_evidence_0")
    with pytest.raises(ValueError, match="evidence copies"):
        verify_export_qualification(case, row["record_id"])


def test_static_snapshot_cache_reuses_material_and_rechecks_changed_assets(tmp_path, source, monkeypatch):
    from . import baseline_cases
    rp, _ = request(tmp_path, source)
    manifest_path = export_cases(rp, tmp_path / "out", synthetic=True)
    row = json.loads(manifest_path.read_text())["records"][0]
    case_path = manifest_path.parent / row["case_path"]
    material_module = importlib.import_module(baseline_cases.PREFIX + ".v21_material")
    original_load, original_sha = material_module.load_material, baseline_cases.sha256
    calls, hashes = [], []
    def count_load(*args, **kwargs):
        calls.append(str(args[0])); return original_load(*args, **kwargs)
    def count_sha(path):
        hashes.append(str(Path(path).resolve())); return original_sha(path)
    monkeypatch.setattr(material_module, "load_material", count_load)
    monkeypatch.setattr(baseline_cases, "sha256", count_sha)
    static, verification = {}, {}
    a = load_baseline_case(case_path, static_cache=static, asset_verification_cache=verification)
    verify_export_qualification(a, row["record_id"], asset_verification_cache=verification)
    b = load_baseline_case(case_path, static_cache=static, asset_verification_cache=verification)
    verify_export_qualification(b, row["record_id"], asset_verification_cache=verification)
    assert len(calls) == 1 and a["blocks"] is b["blocks"]
    material_path = a["asset_receipts"]["material"]["path"]
    assert hashes.count(material_path) == 1
    assert hashes.count(a["asset_receipts"]["geometry"]["path"]) == 1
    assert a["material_cache_moments"] is not b["material_cache_moments"]
    with open(material_path, "r+b") as stream:
        stream.seek(-1, 2); old = stream.read(1)
        stream.seek(-1, 2); stream.write(bytes([old[0] ^ 1]))
    with pytest.raises(ValueError, match="material asset SHA256"):
        load_baseline_case(case_path, static_cache=static, asset_verification_cache=verification)


def test_manufactured_native_contract_portable_roundtrip_is_only_asset_interface(tmp_path, source):
    """Manufactured native contract, NOT teacher evidence or Cherchi validation.

    Numeric labels and mesh come unchanged from the analytic software fixture.
    Only declared identities are changed to exercise the real import interface;
    no derivative is admitted and no native solver or Boolean is executed.
    """
    from .handoff_authority import (
        SCHEMA as AUTHORITY_SCHEMA, STATUS, TEACHER_MAP_PROFILE, canonical_sha256,
        match_authorized_record, verify_packaged_authority,
    )
    raw = tmp_path / "raw"
    raw.mkdir()
    rp, payload = request(raw, source, qualified=False)
    row = payload["records"][0]
    for name in ("geometry", "operand"):
        dest = raw / (name + ".npz")
        shutil.copyfile(row["assets"][name]["path"], dest)
        row["assets"][name]["path"] = str(dest)
    # Reuse the existing static cache to keep this an interface regression.
    material = raw / "material.npz"
    shutil.copyfile(source.parent / "material.npz", material)
    row["assets"]["material"] = dict(path=str(material), sha256=sha256(material))
    meta = row["meta"]
    meta.update(split="train", reference_kind="deployed_native",
                teacher_version="manufactured-native-contract-software-test-only")
    # Preserve all original numeric teacher arrays; qualification remains false.
    with np.load(source.parent / "targets.npz", allow_pickle=False) as original:
        original_arrays = {key: original[key].copy() for key in original.files if key != "metadata"}
    target = Path(row["assets"]["targets"]["path"])
    np.savez(target, **original_arrays,
             metadata=np.asarray(json.dumps({key: meta[key] for key in RAW_IDENTITY_FIELDS})))
    row["assets"]["targets"]["sha256"] = sha256(target)
    qp = Path(row["assets"]["qualification"]["path"])
    qualification = json.loads(qp.read_text())
    qualification["identity"] = {key: meta[key] for key in RAW_IDENTITY_FIELDS}
    qualification["assets_sha256"] = {key: row["assets"][key]["sha256"]
                                        for key in ("geometry", "operand", "targets")}
    rewrite(qp, qualification)
    row["assets"]["qualification"]["sha256"] = sha256(qp)
    support = {}
    for role in ("source_manifest", "teacher_runtime", "teacher_config"):
        file = raw / (role + ".txt")
        file.write_text("Manufactured contract for SOFTWARE INTERFACE TEST ONLY: " + role)
        support[role] = dict(path=str(file), sha256=sha256(file))
    teacher = dict(teacher_version=meta["teacher_version"], backend="cherchi", map_profile=TEACHER_MAP_PROFILE,
                   level=2, growth_policy="GROW_TO_CERTIFIED_V1",
                   runtime_sha256=support["teacher_runtime"]["sha256"],
                   config_sha256=support["teacher_config"]["sha256"])
    teacher_hash = canonical_sha256(teacher)
    payload["snapshot_id"] = "manufactured-contract-software-interface-only"
    authority = dict(schema=AUTHORITY_SCHEMA, snapshot_id=payload["snapshot_id"],
                     teacher_contract=teacher, assets=support, evidence=qualification["evidence"],
                     geometries=[{key: meta[key] for key in ("geometry_id", "source_group", "split", "geometry_sha256")}],
                     admitted_records=[dict(record_id=row["record_id"], identity=qualification["identity"],
                                            assets_sha256=qualification["assets_sha256"],
                                            teacher_contract_sha256=teacher_hash)])
    ap = raw / "authority.json"
    rewrite(ap, authority)
    payload["authority"] = dict(path=str(ap), sha256=sha256(ap))
    payload["teacher_contract_sha256"] = teacher_hash
    rewrite(rp, payload)
    exported = export_cases(rp, tmp_path / "original_output")
    relocated = tmp_path / "relocated_output"
    exported.parent.rename(relocated)
    raw.rename(tmp_path / "unavailable_original_paths")
    manifest_path = relocated / "training_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    portable = manifest["source_authority"]
    bound = verify_packaged_authority(
        relocated / portable["path"], portable["supporting_assets"], manifest["snapshot_id"],
        expected_authority_sha256=portable["sha256"], asset_root=relocated,
    )
    assert bound["receipt"]["status"] == STATUS
    assert bound["receipt"]["independent_teacher_or_split_certification"] is False
    exported_row = manifest["records"][0]
    case = load_baseline_case(relocated / exported_row["case_path"])
    case_assets = case["asset_receipts"]
    hashes = dict(geometry=case_assets["geometry"]["sha256"], operand=case_assets["operand"]["sha256"],
                  targets=case_assets["source_teacher_targets"]["sha256"])
    match_authorized_record(bound, exported_row["record_id"], exported_row["role"], case["meta"], hashes)
    assert hashes == qualification["assets_sha256"]
    assert np.array_equal(case["native_moments"], original_arrays["native_moments"])
    assert not case["meta"]["derivative_valid"] and "native_jacobian" not in case
    qualification_receipt = verify_export_qualification(case, exported_row["record_id"])
    assert qualification_receipt["independent_teacher_or_neighbourhood_reaudit"] is False
