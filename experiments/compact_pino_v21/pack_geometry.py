"""Prepare the two query-independent material controls in solver target coordinates."""
import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np

from research.NeuralMoments.models.pcraft_ufno.candidate_projects.quadrann_reproduction.spin_adapter.v21_material import (
    prepare_material, prepare_rle_material, signed_mesh_moments,
)
from research.NeuralMoments.models.pcraft_ufno.candidate_projects.quadrann_reproduction.spin_adapter.v21_blocks import BlockMeasure
from .diagnostics import moment_errors


def pack(root, indexes, gid):
    source = root / 'geometries' / (gid+'.npz')
    provenance = 'registered native256'
    diagnostic_manifest = root / 'diagnostic256' / 'manifest.json'
    if diagnostic_manifest.exists():
        records = json.loads(diagnostic_manifest.read_text())['rows']
        choices = [r for r in records if r['geometry_id'] == gid]
        if choices:
            ref = choices[0]['target_geometry']
            source = diagnostic_manifest.parent / gid / 'geometry.npz'
            if hashlib.sha256(source.read_bytes()).hexdigest() != ref['sha256']:
                raise ValueError('Work A diagnostic native256 asset hash mismatch')
            provenance = 'Work A direct 256 ray re-encoding on original registered mesh; auxiliary legacy cross-source cohort'
    with np.load(source, allow_pickle=False) as data:
        v, f, columns, origin = (data[k] for k in ('vertices', 'faces', 'columns', 'origin'))
        h, side, rho = float(data['pitch']), int(data['side']), float(data['density'])
        fixed = data['fixed_moments']
        sha = str(data['rle_sha256'])
    if side != 256:
        raise ValueError('a matched, registered native256 grid is required for the RLE control')
    ref = signed_mesh_moments(v, f, rho)
    length = float(np.ptp(v, axis=0).max())
    records = dict(geometry=gid, geometry_export_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
                   grid_provenance=provenance, rle_sha256=sha, origin=origin.tolist(), pitch=h, side=side, density=rho,
                   fixed_moments=fixed.tolist(), reference_editable_mesh=ref.tolist(), length=length, controls={})
    where = indexes / gid
    where.mkdir(parents=True, exist_ok=True)
    for kind in ('rle', 'conservative'):
        t = time.perf_counter()
        mat = (prepare_rle_material(columns, origin, h, side, rho) if kind == 'rle' else
               prepare_material(v, f, origin, h, side, rho))
        material_s = time.perf_counter()-t
        t = time.perf_counter()
        item = where / kind
        reused = item.exists()
        b = BlockMeasure.build(mat, item)
        error = moment_errors(b.moments.sum(0), ref, ref[0], length)
        records['controls'][kind] = dict(identity=mat.identity, full_cells=mat.full_cell_count,
                partial_cells=mat.partial_cell_count, nodes=len(b.nodes), blocks=len(b.ids), full_error=error,
                independent_material_implementation_delta=float(np.max(np.abs(mat.moments() - b.moments.sum(0)))),
                full_editable=b.moments.sum(0).tolist(), material_s=material_s,
                material_audit=getattr(mat, 'audit', {}),
                block_index_s=time.perf_counter()-t, block_index_reused=reused)
        (where/'identity.json').write_text(json.dumps(records, indent=2, allow_nan=False)+'\n')
        print(gid, kind, len(b.nodes), error['scaled_max'], material_s, flush=True)


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--native-root', type=Path, required=True)
    ap.add_argument('--indexes', type=Path, required=True)
    ap.add_argument('geometry', nargs='+')
    args = ap.parse_args()
    for gid in args.geometry:
        pack(args.native_root, args.indexes, gid)
