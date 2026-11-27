"""Frozen training/development case inventory with explicit role isolation."""
from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path

import numpy as np

from ..p0_diagnostics.baseline_cases import (load_baseline_case, moment_scale, sha256, verify_file_receipt,
                                            _native_resolutions, audit_prepared_material)
from ..p0_diagnostics.deterministic_quadrature import _audit_blocks
from ..p0_diagnostics.export_cases import verify_export_qualification
from ..p0_diagnostics.handoff_authority import verify_packaged_authority, match_authorized_record
from .residual_matching import validate_residual_target

SCHEMA = 'compact_pino.training-manifest.v1'


@dataclass
class TrainingData:
    path: Path
    manifest_sha256: str
    snapshot_id: str
    records: list[dict]
    train: list[dict]
    development: list[dict]
    inventory: dict
    physics_receipts: dict
    scope: str
    source_admission: dict | None = None


def _weight(value, name):
    if isinstance(value, bool) or not isinstance(value, (float,int)) or not np.isfinite(value) or value < 0:
        raise ValueError(f'{name} must be finite and nonnegative')
    return float(value)


def _load_residual_target(item, parent, case, mode, verification_cache):
    """Load an optional, explicitly case-bound physical target after split checks.

    Native Moment/J qualification is unchanged. A supplied residual target is a
    separate piece of teacher evidence, never inferred from those two labels.
    """
    inline, asset = item.get('residual_target'), item.get('residual_target_asset')
    if inline is not None and asset is not None:
        raise ValueError('supply residual_target or residual_target_asset, not both')
    target = inline
    if asset is not None:
        if not isinstance(asset, dict):
            raise ValueError('residual_target_asset must contain path and sha256')
        path = Path(asset['path'])
        path = path if path.is_absolute() else parent / path
        receipt = verify_file_receipt(path, asset.get('sha256'), cache=verification_cache,
                                      name='residual target')
        target = json.loads(path.read_text())
        item['residual_target_receipt'] = receipt
    if target is None:
        return False
    if not isinstance(target, dict) or type(target.get('valid')) is not bool:
        raise ValueError('residual target needs an explicit valid Boolean')
    if not target['valid']:
        if not isinstance(target.get('reason'), str) or not target['reason'].strip():
            raise ValueError('invalid residual target requires an explicit reason')
        item['residual_target'] = target
        return False
    binding = target.get('binding')
    if not isinstance(binding, dict) or binding.get('case_sha256') != item['case_sha256']:
        raise ValueError('residual target must bind the exact case manifest SHA256')
    for name, expected in (('record_id', item['record_id']),
                           ('query_id', case['meta']['query_id']),
                           ('geometry_sha256', case['meta']['geometry_sha256'])):
        if name in binding and binding[name] != expected:
            raise ValueError('residual target binding mismatch: ' + name)
    if (mode == 'smoke') != (target.get('teacher_role') == 'synthetic_fixture'):
        raise ValueError('synthetic and native residual targets cannot be interchanged')
    validate_residual_target(target, case['native_moments'], coefficients=case['coefficients'])
    item['residual_target'] = target
    return True


def load_training_manifest(path, *, mode, cache_directory=None,
                            allowed_native_resolutions=(256,), material_layout="blocks"):
    """Read labels only after role and split checks; never relabel a case's role.

    ``loader_eligible`` only controls membership. It is not authorization to
    train and is not evidence of a successful geometric derivative audit.
    """
    if mode not in ('smoke', 'trial', 'long'):
        raise ValueError('mode must be smoke, trial or long')
    allowed_native_resolutions = _native_resolutions(allowed_native_resolutions)
    if material_layout not in ("blocks", "prepared"):
        raise ValueError("material_layout must be blocks or prepared")
    path = Path(path).resolve(strict=True)
    raw = json.loads(path.read_text())
    if raw.get('schema') != SCHEMA or not str(raw.get('snapshot_id','')).strip():
        raise ValueError('a frozen training-manifest.v1 and snapshot_id are required')
    if 'export_receipt_sha256' in raw:
        receipt_path = Path(raw.get('export_receipt_path','export_receipt.json'))
        receipt_path = receipt_path if receipt_path.is_absolute() else path.parent/receipt_path
        if not receipt_path.is_file() or sha256(receipt_path) != raw['export_receipt_sha256']:
            raise ValueError('export receipt SHA256 mismatch')
    authority = None
    if mode != 'smoke':
        declared = raw.get('source_authority')
        if not isinstance(declared,dict):
            raise ValueError('real training requires the frozen external source_authority admission handoff')
        authority_path = Path(declared['path'])
        authority_path = authority_path if authority_path.is_absolute() else path.parent/authority_path
        authority = verify_packaged_authority(authority_path,declared['supporting_assets'],raw['snapshot_id'],
            expected_authority_sha256=declared['sha256'],asset_root=path.parent)
        if declared.get('teacher_contract_sha256') != authority['receipt']['teacher_contract_sha256']:
            raise ValueError('manifest teacher contract differs from frozen source authority')
    rows = raw.get('records')
    if not isinstance(rows,list) or not rows:
        raise ValueError('nonempty manifest records required')
    expected_scope = 'synthetic' if mode == 'smoke' else 'development_training'
    loaded, inventory_rows, seen_ids, seen_queries = [], [], set(), set()
    memberships = {'train': {'sources':set(), 'geometries':set(), 'geometry_hashes':set()},
                   'development': {'sources':set(), 'geometries':set(), 'geometry_hashes':set()}}
    physics_receipts = {}
    for row in rows:
        if not isinstance(row,dict) or row.get('role') not in ('train','development'):
            raise ValueError('only train/development roles are allowed; test/benchmark is forbidden')
        rid = row.get('record_id')
        if not isinstance(rid,str) or not rid.strip() or rid in seen_ids:
            raise ValueError('record_id must be nonempty and unique')
        seen_ids.add(rid)
        if type(row.get('loader_eligible')) is not bool:
            raise ValueError('explicit loader_eligible Boolean required')
        candidate = Path(row['case_path'])
        candidate = candidate if candidate.is_absolute() else path.parent/candidate
        candidate = candidate.resolve(strict=True)
        if sha256(candidate) != row.get('case_sha256'):
            raise ValueError(f'case manifest SHA256 mismatch: {rid}')
        # The small manifest is read before any targets. Ineligible cases remain
        # in the declared inventory but their target assets are never opened.
        case_spec = json.loads(candidate.read_text())
        case_meta = case_spec['meta']
        split = case_meta.get('split')
        if split in ('test','locked_test','fixed_benchmark','benchmark'):
            raise ValueError('test/benchmark input is forbidden even when loader-ineligible')
        if mode == 'smoke':
            if split != 'synthetic' or case_meta.get('reference_kind') != 'analytic_fixture':
                raise ValueError('smoke accepts explicitly synthetic analytic fixtures only')
        elif split != row['role'] or case_meta.get('reference_kind') != 'deployed_native':
            raise ValueError('real training/development roles must match deployed-native case split')
        if authority is not None:
            assets = case_spec['assets']
            hashes = {key:assets[key]['sha256'] for key in ('geometry','operand')}
            hashes['targets'] = case_spec['source_teacher_targets']['sha256']
            match_authorized_record(authority,rid,row['role'],case_meta,hashes)
        query_key = (case_meta['geometry_id'],case_meta['query_id'])
        if query_key in seen_queries:
            raise ValueError('duplicate geometry/query identity in frozen manifest')
        seen_queries.add(query_key)
        memberships[row['role']]['sources'].add(case_meta['source_group'])
        memberships[row['role']]['geometries'].add(case_meta['geometry_id'])
        memberships[row['role']]['geometry_hashes'].add(case_meta['geometry_sha256'])
        qualified = case_meta.get('derivative_valid')
        if type(qualified) is not bool:
            raise ValueError('case derivative qualification must be explicit')
        dimension = case_meta.get('derivative_dimension')
        derivative_type = {3:'R3',8:'R8',10:'Full-J'}.get(dimension)
        item = dict(row, case_path=str(candidate), case_meta=case_meta,
                    moment_weight=_weight(row.get('moment_weight',1.),'moment_weight'),
                    jacobian_weight=_weight(row.get('jacobian_weight',1.),'jacobian_weight'),
                    residual_weight=_weight(row.get('residual_weight',row.get('moment_weight',1.)),
                                            'residual_weight'))
        if row['role'] == 'train':
            inventory_rows.append(dict(record_id=rid, query_id=case_meta['query_id'],
                geometry_id=case_meta['geometry_id'], source_group_id=case_meta['source_group'],
                moment_valid=True, jacobian_valid=qualified, loader_eligible=row['loader_eligible'],
                derivative_type=derivative_type, derivative_dimension=dimension,
                chart_id=case_meta.get('chart_id'),
                **{key:row.get(key) for key in ('surface_family','retention_bin','trajectory_stage')}))
        loaded.append(item)
    for field in ('sources','geometries','geometry_hashes'):
        if memberships['train'][field] & memberships['development'][field]:
            raise ValueError(f'training and development {field} must be disjoint')
    # Only after checking ALL metadata/source memberships do we open a target.
    # Share one memmap per material rather than holding one mapping per query.
    shared_blocks,static_cache,asset_verification_cache = {},{},{}
    inventory_by_id = {row['record_id']: row for row in inventory_rows}
    for item in loaded:
        if not item['loader_eligible']:
            continue
        meta = item['case_meta']
        resolution_kwargs = ({} if tuple(allowed_native_resolutions) == (256,) else
                             dict(allowed_native_resolutions=allowed_native_resolutions))
        if material_layout != "blocks":
            resolution_kwargs["material_layout"] = material_layout
        case = load_baseline_case(item['case_path'], cache_directory=cache_directory,
                                 static_cache=static_cache,asset_verification_cache=asset_verification_cache,
                                 **resolution_kwargs)
        if mode != 'smoke':
            qualification = verify_export_qualification(case,item['record_id'],
                asset_verification_cache=asset_verification_cache)
            if qualification is None:
                raise ValueError('real cases require the exported original-teacher binding receipt; unqualified J must remain Moment-only')
            item['qualification_receipt'] = qualification
        scale = moment_scale(meta['editable_mass'],meta['length_scale'])
        key = (f"{meta['material_identity']}/prepared" if material_layout == "prepared" else
               f"{meta['material_identity']}/blocks-{case['blocks'].block_voxels}")
        if key not in physics_receipts:
            audit = audit_prepared_material if material_layout == "prepared" else _audit_blocks
            physics_receipts[key] = audit(case['blocks'],scale,case['material_cache_moments'])
            shared_blocks[key] = case['blocks']
        else:
            case['blocks'] = shared_blocks[key]
        full = np.asarray(physics_receipts[key]['cubature_moments'])
        error = float(np.max(np.abs(full-case['native_full_editable_moments'])/scale))
        if error > 1e-8:
            raise ValueError(f"full material ten-moment gate failed for {item['record_id']}: {error}")
        residual_valid = _load_residual_target(item, path.parent, case, mode, asset_verification_cache)
        item['residual_valid'] = residual_valid
        if item['role'] == 'train':
            inventory_by_id[item['record_id']]['residual_valid'] = residual_valid
        if (item['moment_weight'] == 0 and (not meta['derivative_valid'] or item['jacobian_weight'] == 0)
                and not (residual_valid and item['residual_weight'] > 0)):
            raise ValueError(f"loader-eligible row has no admitted positive-weight objective: {item['record_id']}")
        item['case'] = case
    train = [row for row in loaded if row['role']=='train' and row['loader_eligible']]
    development = [row for row in loaded if row['role']=='development' and row['loader_eligible']]
    if not train:
        raise ValueError('no eligible training cases')
    registered = raw.get('registered_train_count')
    if type(registered) is not int or registered < len(inventory_rows):
        raise ValueError('registered_train_count must cover all declared training records')
    digest = sha256(path)
    inventory = dict(snapshot_id=raw['snapshot_id'],manifest_sha256=digest,
                     registered_count=registered,records=inventory_rows)
    return TrainingData(path,digest,raw['snapshot_id'],loaded,train,development,inventory,
                        physics_receipts,expected_scope,authority['receipt'] if authority else None)
