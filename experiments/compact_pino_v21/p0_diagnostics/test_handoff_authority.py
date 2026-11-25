"""Software-only declarations exercise handoff consistency, not native physics."""
import copy
import hashlib
import json
from pathlib import Path
import shutil

import pytest

from .baseline_cases import COEFFICIENT_MULTIPLIERS, ORDER
from .handoff_authority import (
    RAW_IDENTITY_FIELDS, SCHEMA, STATUS, TEACHER_MAP_PROFILE, canonical_sha256,
    match_authorized_record, verify_handoff_authority, verify_packaged_authority,
)


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _write(path, value):
    Path(path).write_text(json.dumps(value), encoding="utf8")


def _asset(folder, name, value):
    path = folder / name
    path.write_text(value, encoding="utf8")
    return dict(path=str(path), sha256=_sha(path))


@pytest.fixture
def declaration(tmp_path):
    raw = tmp_path / "raw"
    raw.mkdir()
    support = {name: _asset(raw, name + ".txt", "SOFTWARE TEST ONLY: " + name)
               for name in ("source_manifest", "teacher_runtime", "teacher_config")}
    evidence = [_asset(raw, "extra_evidence.txt", "software declaration only")]
    teacher = dict(teacher_version="software-test-finite-teacher", backend="cherchi",
                   map_profile=TEACHER_MAP_PROFILE, level=2,
                   growth_policy="GROW_TO_CERTIFIED_V1",
                   runtime_sha256=support["teacher_runtime"]["sha256"],
                   config_sha256=support["teacher_config"]["sha256"])
    contract_hash = canonical_sha256(teacher)
    registry, admitted, rows = [], [], []
    for gid, group, split in (("R01", "source_1", "train"), ("R02", "source_2", "development")):
        assets = {name: _asset(raw, gid + "_" + name + ".txt", gid + ":" + name)
                  for name in ("geometry", "operand", "targets")}
        meta = dict(geometry_id=gid, query_id="query_1", source_group=group, split=split,
                    teacher_version=teacher["teacher_version"], coordinate_frame="solver",
                    units="m_kg", moment_order=list(ORDER), coefficient_basis="raw_coefficients",
                    coefficient_multipliers=list(COEFFICIENT_MULTIPLIERS),
                    geometry_sha256=assets["geometry"]["sha256"],
                    operand_content_sha256="b" * 64, query_content_sha256="c" * 64,
                    operation="intersection", reference_kind="deployed_native",
                    moments_include_fixed=True, derivative_valid=False,
                    derivative_dimension=None, chart_id="unqualified",
                    derivative_reason="software_fixture_not_qualified", length_scale=1.,
                    length_scale_definition="editable_bbox_max_extent", editable_mass=1., density=1.)
        registry.append(dict(geometry_id=gid, source_group=group, split=split,
                             geometry_sha256=assets["geometry"]["sha256"]))
        rid = gid + "/query_1"
        admitted.append(dict(record_id=rid, identity={key: meta[key] for key in RAW_IDENTITY_FIELDS},
                             assets_sha256={key: item["sha256"] for key, item in assets.items()},
                             teacher_contract_sha256=contract_hash))
        rows.append(dict(record_id=rid, role=split, meta=meta, assets=assets))
    registry.append(dict(geometry_id="locked_shape", source_group="locked_source",
                         split="locked_test", geometry_sha256="d" * 64))
    authority = dict(schema=SCHEMA, snapshot_id="392-test-declaration", teacher_contract=teacher,
                     assets=support, evidence=evidence, geometries=registry, admitted_records=admitted)
    ap = raw / "authority.json"
    _write(ap, authority)
    request = dict(schema="tvcg.v22.case-export.v1", snapshot_id=authority["snapshot_id"],
                   authority=dict(path=str(ap), sha256=_sha(ap)),
                   teacher_contract_sha256=contract_hash, records=rows)
    rp = raw / "request.json"
    _write(rp, request)
    return rp, request, authority


def _save(request_path, request, authority=None):
    if authority is not None:
        path = Path(request["authority"]["path"])
        _write(path, authority)
        request["authority"]["sha256"] = _sha(path)
    _write(request_path, request)


def test_real_subset_is_bound_to_full_registry(declaration):
    path, request, _ = declaration
    request["records"] = request["records"][1:]  # The omitted training geometry stays in the registry.
    _save(path, request)
    verified = verify_handoff_authority(request, path)
    receipt = verified["receipt"]
    assert receipt["status"] == STATUS
    assert receipt["scope"] == "external_frozen_admission_handoff_only"
    assert receipt["counts"] == dict(geometries=3, admitted_records=2,
                                      records_by_split={"train": 1, "development": 1})
    assert receipt["requested_records"] == 1
    assert receipt["independent_teacher_or_split_certification"] is False
    assert set(receipt["verified_assets"]) == {"source_manifest", "teacher_runtime", "teacher_config", "evidence_0000"}


def test_development_cannot_be_relabeled_in_a_partial_request(declaration):
    path, request, _ = declaration
    row = request["records"][1]
    request["records"] = [row]
    row["role"] = row["meta"]["split"] = "train"
    row["assets"]["targets"]["path"] = "/never/open/this/label.npz"
    _save(path, request)
    with pytest.raises(ValueError, match="frozen split"):
        verify_handoff_authority(request, path)


def test_all_memberships_checked_before_any_query_file(declaration):
    path, request, _ = declaration
    request["records"][0]["assets"]["targets"]["path"] = "/never/open/first.npz"
    request["records"][1]["record_id"] = "unknown/query"
    _save(path, request)
    with pytest.raises(ValueError, match="absent from frozen"):
        verify_handoff_authority(request, path)


@pytest.mark.parametrize("mutation, expected", [
    ("role", "locked/diagnostic"),
    ("identity", "RAW identity"),
    ("targets", "requested asset"),
    ("teacher", "teacher contract"),
    ("row_teacher", "different teacher"),
    ("snapshot", "snapshot_id"),
    ("duplicate", "duplicate requested"),
])
def test_request_binding_rejects_drift(declaration, mutation, expected):
    path, request, _ = declaration
    if mutation == "role":
        request["records"][0]["role"] = "locked_test"
    elif mutation == "identity":
        request["records"][0]["meta"]["density"] = 2.
    elif mutation == "targets":
        request["records"][0]["assets"]["targets"]["sha256"] = "a" * 64
    elif mutation == "teacher":
        request["teacher_contract_sha256"] = "a" * 64
    elif mutation == "row_teacher":
        request["records"][0]["teacher_contract_sha256"] = "a" * 64
    elif mutation == "snapshot":
        request["snapshot_id"] = "different"
    else:
        request["records"].append(copy.deepcopy(request["records"][0]))
    _save(path, request)
    with pytest.raises(ValueError, match=expected):
        verify_handoff_authority(request, path)


@pytest.mark.parametrize("mutation, expected", [
    ("group_split", "source group straddles"),
    ("geometry_alias", "geometry content straddles"),
    ("registry_role", "frozen geometry split"),
    ("duplicate_geometry", "duplicate authority geometry"),
    ("duplicate_record", "duplicate authority record"),
    ("mixed_teacher", "different teacher contract"),
    ("runtime", "runtime identity"),
    ("map", "finite Cherchi"),
    ("level", "level=2"),
    ("growth", "GROW_TO_CERTIFIED_V1"),
])
def test_global_authority_is_consistent_even_outside_requested_subset(declaration, mutation, expected):
    path, request, authority = declaration
    request["records"] = request["records"][:1]
    if mutation == "group_split":
        authority["geometries"][1]["source_group"] = authority["geometries"][0]["source_group"]
    elif mutation == "geometry_alias":
        authority["geometries"][1]["geometry_sha256"] = authority["geometries"][0]["geometry_sha256"]
    elif mutation == "registry_role":
        authority["geometries"][1]["split"] = "locked_test"
    elif mutation == "duplicate_geometry":
        authority["geometries"].append(copy.deepcopy(authority["geometries"][1]))
    elif mutation == "duplicate_record":
        authority["admitted_records"].append(copy.deepcopy(authority["admitted_records"][1]))
    elif mutation == "mixed_teacher":
        authority["admitted_records"][1]["teacher_contract_sha256"] = "a" * 64
    elif mutation == "runtime":
        authority["teacher_contract"]["runtime_sha256"] = "a" * 64
    elif mutation == "map":
        authority["teacher_contract"]["backend"] = "cgal"
    elif mutation == "level":
        authority["teacher_contract"]["level"] = 3
    elif mutation == "growth":
        authority["teacher_contract"]["growth_policy"] = "some_other_policy"
    _save(path, request, authority)
    with pytest.raises(ValueError, match=expected):
        verify_handoff_authority(request, path)


def test_actual_bytes_must_match_authority_and_query_hashes(declaration):
    path, request, _ = declaration
    target = Path(request["records"][0]["assets"]["targets"]["path"])
    target.write_text("changed")
    with pytest.raises(ValueError, match="query targets SHA256"):
        verify_handoff_authority(request, path)


def test_portable_copy_ignores_original_absolute_paths(declaration, tmp_path, monkeypatch):
    path, request, _ = declaration
    original = verify_handoff_authority(request, path)
    packed = tmp_path / "packed"
    packed.mkdir()
    ap = packed / "authority.json"
    shutil.copyfile(request["authority"]["path"], ap)
    copies = {}
    for role, receipt in original["receipt"]["verified_assets"].items():
        destination = packed / (role + ".asset")
        shutil.copyfile(receipt["path"], destination)
        copies[role] = dict(path=destination.name, sha256=receipt["sha256"])
    path.parent.rename(tmp_path / "removed_original_paths")
    verified = verify_packaged_authority(ap, copies, request["snapshot_id"],
                                         expected_authority_sha256=request["authority"]["sha256"])
    assert verified["record_table"] == original["record_table"]
    def no_io(*args, **kwargs):
        raise AssertionError("per-record authorization must perform no I/O")
    monkeypatch.setattr(Path, "open", no_io)
    row = request["records"][0]
    assert match_authorized_record(verified, row["record_id"], row["role"], row["meta"],
                                   {k: v["sha256"] for k, v in row["assets"].items()})["record_id"] == row["record_id"]


@pytest.mark.parametrize("mutation", ["missing", "extra", "changed_hash", "changed_bytes"])
def test_portable_support_set_cannot_change(declaration, mutation):
    path, request, authority = declaration
    assets = dict(authority["assets"], evidence_0000=authority["evidence"][0])
    if mutation == "missing":
        assets.pop("teacher_runtime")
    elif mutation == "extra":
        assets["arbitrary"] = assets["teacher_runtime"]
    elif mutation == "changed_hash":
        assets["teacher_runtime"] = dict(assets["teacher_runtime"], sha256="a" * 64)
    else:
        Path(assets["teacher_runtime"]["path"]).write_text("changed runtime")
    with pytest.raises(ValueError, match="supporting asset|SHA256"):
        verify_packaged_authority(request["authority"]["path"], assets, request["snapshot_id"],
                                  expected_authority_sha256=request["authority"]["sha256"])


def test_synthetic_is_explicitly_outside_real_authority(declaration):
    path, request, _ = declaration
    for row in request["records"]:
        row["meta"].update(split="synthetic", reference_kind="analytic_fixture")
    request.pop("authority")
    request.pop("teacher_contract_sha256")
    _save(path, request)
    verified = verify_handoff_authority(request, path)
    assert verified["receipt"]["status"] == "NOT_APPLICABLE_SYNTHETIC"
    assert verified["record_table"] == {}
    with pytest.raises(ValueError, match="real record requires"):
        match_authorized_record(verified, "anything", "train", {}, {})
    request["records"][0]["meta"].update(split="train", reference_kind="deployed_native")
    _save(path, request)
    with pytest.raises(ValueError, match="mixed or mislabeled"):
        verify_handoff_authority(request, path)


def test_duplicate_json_keys_and_unfrozen_request_are_rejected(declaration):
    path, request, _ = declaration
    request["snapshot_id"] = "in-memory-only-change"
    with pytest.raises(ValueError, match="frozen request file"):
        verify_handoff_authority(request, path)
    ap = Path(request["authority"]["path"])
    ap.write_text('{"schema":"old","schema":"new"}')
    with pytest.raises(ValueError, match="duplicate JSON key"):
        verify_packaged_authority(ap, {}, request["snapshot_id"], expected_authority_sha256=_sha(ap))
