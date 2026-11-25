"""Bind an imported subset to Work A's external frozen admission declaration.

An authority binds the full geometry/source split registry, admitted raw query
identities, a single deployed finite Cherchi teacher contract, and the original
manifest/runtime/config artifacts. These are consistency checks on a supplied
external declaration, not independent certification of its scientific claims.

``verify_handoff_authority`` checks an export request and its original assets.
``verify_packaged_authority`` checks immutable authority bytes against portable
copies, without opening the authority's old absolute paths. Call it once per
manifest; ``match_authorized_record`` then performs no file I/O per query.
"""
from __future__ import annotations

from collections import Counter
import hashlib
import json
from pathlib import Path

from .baseline_cases import LABEL_IDENTITY_FIELDS, validate_meta

SCHEMA = "tvcg.v22.handoff-authority.v1"
REQUEST_SCHEMA = "tvcg.v22.case-export.v1"
RAW_IDENTITY_FIELDS = tuple(key for key in LABEL_IDENTITY_FIELDS if key != "material_identity")
TEACHER_MAP_PROFILE = "finite_directed_capped_intersection_difference"
SCOPE = "external_frozen_admission_handoff_only"
STATUS = "BOUND_EXTERNAL_ADMISSION_DECLARATION_VERIFIED"
ASSET_ROLES = ("source_manifest", "teacher_runtime", "teacher_config")
QUERY_ASSETS = ("geometry", "operand", "targets")


def _object(value, name):
    if not isinstance(value, dict):
        raise ValueError(name + " must be an object")
    return value


def _text(value, name):
    if not isinstance(value, str) or not value.strip():
        raise ValueError("nonempty " + name + " required")
    return value


def _digest(value, name):
    if not isinstance(value, str) or len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
        raise ValueError(name + " must be a lowercase SHA256")
    return value


def canonical_sha256(value):
    """Hash the complete JSON teacher contract, independent of formatting."""
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     ensure_ascii=False, allow_nan=False).encode("utf8")).hexdigest()


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate JSON key: " + key)
        result[key] = value
    return result


def _read(path):
    def invalid(value):
        raise ValueError("nonfinite JSON value: " + value)
    return _object(json.loads(Path(path).read_text(encoding="utf8"),
                              object_pairs_hook=_unique_object, parse_constant=invalid), "document")


def _checked_asset(entry, root, name, cache):
    entry = _object(entry, name)
    expected = _digest(entry.get("sha256"), name + " sha256")
    path = Path(_text(entry.get("path"), name + " path"))
    path = (path if path.is_absolute() else root / path).resolve(strict=True)
    if path not in cache:
        with path.open("rb") as stream:
            cache[path] = hashlib.file_digest(stream, "sha256").hexdigest()
    if cache[path] != expected:
        raise ValueError(name + " SHA256 mismatch")
    return dict(path=str(path), sha256=expected, size_bytes=path.stat().st_size)


def _raw_identity(value, name):
    identity = _object(value, name)
    missing = set(RAW_IDENTITY_FIELDS) - identity.keys()
    if missing:
        raise ValueError(name + " missing RAW identity fields: " + ", ".join(sorted(missing)))
    if identity["split"] not in ("train", "development") or identity["reference_kind"] != "deployed_native":
        raise ValueError(name + " must be an admitted train/development deployed_native identity")
    # This placeholder is used only to run existing raw convention validation;
    # material is constructed later. An unqualified flag remains unqualified.
    structural = dict(identity, material_identity="0" * 64)
    structural.setdefault("derivative_reason", "external_authority_declares_unqualified")
    validate_meta(structural)
    return identity


def _tables(authority, expected_snapshot_id):
    if authority.get("schema") != SCHEMA:
        raise ValueError("unsupported handoff authority schema")
    snapshot = _text(authority.get("snapshot_id"), "authority snapshot_id")
    if snapshot != _text(expected_snapshot_id, "expected snapshot_id"):
        raise ValueError("authority snapshot_id mismatch")
    teacher = _object(authority.get("teacher_contract"), "teacher_contract")
    _text(teacher.get("teacher_version"), "teacher_version")
    if teacher.get("backend") != "cherchi" or teacher.get("map_profile") != TEACHER_MAP_PROFILE:
        raise ValueError("authority requires the explicit deployed finite Cherchi map profile")
    if type(teacher.get("level")) is not int or teacher["level"] != 2:
        raise ValueError("this frozen teacher contract requires level=2")
    if teacher.get("growth_policy") != "GROW_TO_CERTIFIED_V1":
        raise ValueError("this frozen teacher contract requires GROW_TO_CERTIFIED_V1")
    for name in ("runtime_sha256", "config_sha256"):
        _digest(teacher.get(name), "teacher " + name)
    teacher_hash = canonical_sha256(teacher)

    original_assets = _object(authority.get("assets"), "authority assets")
    if set(original_assets) != set(ASSET_ROLES):
        raise ValueError("authority assets must contain source_manifest, teacher_runtime and teacher_config")
    expected_assets = dict(original_assets)
    evidence = authority.get("evidence", [])
    if not isinstance(evidence, list):
        raise ValueError("authority evidence must be a list")
    expected_assets.update({f"evidence_{i:04d}": entry for i, entry in enumerate(evidence)})
    for role, entry in expected_assets.items():
        _object(entry, role)
        _text(entry.get("path"), role + " path")
        _digest(entry.get("sha256"), role + " sha256")
    if original_assets["teacher_runtime"]["sha256"] != teacher["runtime_sha256"]:
        raise ValueError("teacher runtime identity differs from authority asset")
    if original_assets["teacher_config"]["sha256"] != teacher["config_sha256"]:
        raise ValueError("teacher config identity differs from authority asset")

    entries = authority.get("geometries")
    if not isinstance(entries, list) or not entries:
        raise ValueError("nonempty frozen geometry registry required")
    geometries, source_splits, content_splits = {}, {}, {}
    for item in entries:
        item = _object(item, "geometry registry entry")
        gid = _text(item.get("geometry_id"), "geometry_id")
        group = _text(item.get("source_group"), "source_group")
        split = _text(item.get("split"), "geometry split")
        digest = _digest(item.get("geometry_sha256"), "geometry_sha256")
        if gid in geometries:
            raise ValueError("duplicate authority geometry_id: " + gid)
        if split == "synthetic":
            raise ValueError("synthetic geometries cannot enter real admission authority")
        if source_splits.setdefault(group, split) != split:
            raise ValueError("frozen source group straddles splits: " + group)
        if content_splits.setdefault(digest, split) != split:
            raise ValueError("frozen geometry content straddles splits: " + gid)
        geometries[gid] = dict(item)

    entries = authority.get("admitted_records")
    if not isinstance(entries, list) or not entries:
        raise ValueError("nonempty frozen admitted_records required")
    records, query_keys = {}, set()
    for item in entries:
        item = _object(item, "admitted record")
        rid = _text(item.get("record_id"), "record_id")
        if rid in records:
            raise ValueError("duplicate authority record_id: " + rid)
        identity = _raw_identity(item.get("identity"), "authority record " + rid)
        key = (identity["geometry_id"], identity["query_id"], identity["chart_id"])
        if key in query_keys:
            raise ValueError("duplicate authority query/chart identity")
        query_keys.add(key)
        geometry = geometries.get(identity["geometry_id"])
        if geometry is None:
            raise ValueError("admitted record references unknown geometry: " + rid)
        for name in ("source_group", "split", "geometry_sha256"):
            if identity[name] != geometry[name]:
                raise ValueError("authority record differs from frozen geometry " + name + ": " + rid)
        if identity["teacher_version"] != teacher["teacher_version"] or item.get("teacher_contract_sha256") != teacher_hash:
            raise ValueError("authority record uses a different teacher contract: " + rid)
        hashes = _object(item.get("assets_sha256"), "admitted assets_sha256")
        if set(hashes) != set(QUERY_ASSETS):
            raise ValueError("admitted assets_sha256 must bind geometry, operand and targets")
        for name in QUERY_ASSETS:
            _digest(hashes[name], "admitted " + name)
        if hashes["geometry"] != identity["geometry_sha256"]:
            raise ValueError("admitted geometry asset differs from raw identity")
        records[rid] = dict(item)
    return teacher, teacher_hash, expected_assets, geometries, records


def _verified(authority, authority_receipt, expected_snapshot_id, *, asset_overrides=None,
              asset_root, cache):
    teacher, teacher_hash, originals, geometries, records = _tables(authority, expected_snapshot_id)
    assets = originals if asset_overrides is None else _object(asset_overrides, "supporting_assets")
    if set(assets) != set(originals):
        raise ValueError("portable supporting asset roles differ from authority")
    for role, entry in assets.items():
        if _object(entry, role).get("sha256") != originals[role]["sha256"]:
            raise ValueError("portable supporting asset SHA256 differs from authority: " + role)
    receipts = {role: _checked_asset(entry, asset_root, role, cache) for role, entry in assets.items()}
    receipt = dict(status=STATUS, scope=SCOPE, snapshot_id=expected_snapshot_id,
                   authority=authority_receipt, verified_assets=receipts,
                   teacher_contract_sha256=teacher_hash,
                   independent_teacher_or_split_certification=False,
                   counts=dict(geometries=len(geometries), admitted_records=len(records),
                               records_by_split=dict(Counter(item["identity"]["split"] for item in records.values()))))
    return dict(receipt=receipt, teacher_contract=teacher,
                geometry_table=geometries, record_table=records)


def match_authorized_record(verified, record_id, role, meta, assets_sha256):
    """Match one real row to verified tables; never reopen large supporting files."""
    if verified.get("receipt", {}).get("status") != STATUS:
        raise ValueError("real record requires a verified external admission declaration")
    rid = _text(record_id, "record_id")
    if role not in ("train", "development"):
        raise ValueError("locked/diagnostic/unknown roles cannot enter training or development")
    frozen = verified["record_table"].get(rid)
    if frozen is None:
        raise ValueError("record is absent from frozen admitted_records: " + rid)
    expected = frozen["identity"]
    if role != expected["split"]:
        raise ValueError("requested role differs from frozen split: " + rid)
    identity = _raw_identity(meta, "requested record " + rid)
    for name in RAW_IDENTITY_FIELDS:
        if identity[name] != expected[name]:
            raise ValueError("requested RAW identity differs from authority: " + name)
    hashes = _object(assets_sha256, "requested assets_sha256")
    for name in QUERY_ASSETS:
        if hashes.get(name) != frozen["assets_sha256"][name]:
            raise ValueError("requested asset differs from authority: " + name)
    return frozen


def verify_packaged_authority(authority_path, copied_asset_receipts, expected_snapshot_id, *,
                              expected_authority_sha256, asset_root=None):
    """Verify portable copies once, ignoring all original paths in the authority."""
    path = Path(authority_path).resolve(strict=True)
    root = path.parent if asset_root is None else Path(asset_root).resolve(strict=True)
    cache = {}
    receipt = _checked_asset(dict(path=str(path), sha256=expected_authority_sha256), root, "authority", cache)
    return _verified(_read(path), receipt, expected_snapshot_id,
                     asset_overrides=copied_asset_receipts, asset_root=root, cache=cache)


def verify_handoff_authority(request, request_path):
    """Validate a real export subset before any label import or material build.

    All row membership checks complete before query assets are opened. Explicit
    synthetic fixtures return NOT_APPLICABLE_SYNTHETIC, never real admission.
    """
    request = _object(request, "request")
    path = Path(request_path).resolve(strict=True)
    if _read(path) != request:
        raise ValueError("request object differs from frozen request file")
    if request.get("schema") != REQUEST_SCHEMA:
        raise ValueError("unsupported export request schema")
    snapshot = _text(request.get("snapshot_id"), "request snapshot_id")
    rows = request.get("records")
    if not isinstance(rows, list) or not rows:
        raise ValueError("nonempty request records required")
    for row in rows:
        _object(_object(row, "request record").get("meta"), "request meta")
    synthetic = [(row["meta"].get("split") == "synthetic"
                  or row["meta"].get("reference_kind") == "analytic_fixture") for row in rows]
    if any(synthetic):
        if not all(synthetic) or any(row.get("meta", {}).get("split") != "synthetic" or
                                     row["meta"].get("reference_kind") != "analytic_fixture" for row in rows):
            raise ValueError("mixed or mislabeled synthetic/real authority request")
        return dict(receipt=dict(status="NOT_APPLICABLE_SYNTHETIC", scope="synthetic_software_only",
                                 snapshot_id=snapshot, independent_teacher_or_split_certification=False),
                    teacher_contract=None, geometry_table={}, record_table={})
    cache = {}
    authority_receipt = _checked_asset(request.get("authority"), path.parent, "authority", cache)
    authority_path = Path(authority_receipt["path"])
    verified = _verified(_read(authority_path), authority_receipt, snapshot,
                         asset_root=authority_path.parent, cache=cache)
    contract_hash = _digest(request.get("teacher_contract_sha256"), "request teacher_contract_sha256")
    if contract_hash != verified["receipt"]["teacher_contract_sha256"]:
        raise ValueError("request teacher contract differs from authority")
    ids = set()
    for row in rows:
        rid = _text(row.get("record_id"), "request record_id")
        if rid in ids:
            raise ValueError("duplicate requested record_id")
        ids.add(rid)
        assets = _object(row.get("assets"), "request assets")
        hashes = {name: _object(assets.get(name), "request " + name).get("sha256") for name in QUERY_ASSETS}
        match_authorized_record(verified, rid, row.get("role"), row.get("meta"), hashes)
        if "teacher_contract_sha256" in row and row["teacher_contract_sha256"] != contract_hash:
            raise ValueError("requested row uses a different teacher contract")
    # Never reach requested labels when any row is unknown, locked, or relabeled.
    for row in rows:
        for name in QUERY_ASSETS:
            _checked_asset(row["assets"][name], path.parent, "query " + name, cache)
    with path.open("rb") as stream:
        request_hash = hashlib.file_digest(stream, "sha256").hexdigest()
    verified["receipt"].update(request_sha256=request_hash, requested_records=len(rows))
    return verified
