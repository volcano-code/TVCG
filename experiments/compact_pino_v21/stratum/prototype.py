"""Run the frozen 256³ deterministic response on geometry-only operands, then score separately."""
import argparse
import gzip
import hashlib
import inspect
import json
import math
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

from experiments.compact_pino_v21.diagnostics import jacobian_errors, moment_errors
from .physical_replay.archive import PATTERN, historic

HERE = Path(__file__).resolve().parent
STATIC = HERE.parent / 'evidence' / 'static_material.json'
ADAPTER=HERE.parents[2]/'research/NeuralMoments/models/pcraft_ufno/candidate_projects/quadrann_reproduction/spin_adapter'


def read(path):
    return json.loads(path.read_bytes())


def write(path,record):
    path.parent.mkdir(parents=True,exist_ok=True)
    stage=path.with_suffix('.tmp')
    stage.write_text(json.dumps(record,indent=1,allow_nan=False)+'\n')
    stage.replace(path)


def digest(path):
    with path.open('rb') as source:
        return hashlib.file_digest(source, 'sha256').hexdigest()


def recipe():
    sources=('compact_pino_v21.py','v21_native.py','v21_blocks.py','v21_response.py','v21_response_surface.py')
    return dict(implementation_sha256={name:digest(ADAPTER/name) for name in sources},
                prediction_entry_sha256=hashlib.sha256(inspect.getsource(predict).encode()).hexdigest(),bandwidth_voxels=1.0,
                spatial_block_width_voxels=8,chunk_size=65536,model=None,device='cpu',dtype='float64')


def recipe_sha(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':')).encode()).hexdigest()


def predict(operands, indexes, output):
    import torch
    from research.NeuralMoments.models.pcraft_ufno.candidate_projects.quadrann_reproduction.spin_adapter.compact_pino_v21 import evaluate_packed
    from research.NeuralMoments.models.pcraft_ufno.candidate_projects.quadrann_reproduction.spin_adapter.v21_blocks import BlockMeasure
    from research.NeuralMoments.models.pcraft_ufno.candidate_projects.quadrann_reproduction.spin_adapter.v21_native import FiniteOperand
    from research.NeuralMoments.models.pcraft_ufno.candidate_projects.quadrann_reproduction.spin_adapter.v21_response import prepare_surface

    torch.set_num_threads(1)
    static = read(STATIC)
    identity_sha=recipe_sha(recipe())
    registered=read(HERE/'protocol.json')['population']
    for gid in sorted(set(registered['R3'])|set(registered['R8'])):
        meta = static[gid]
        local = read(indexes/gid/'identity.json')
        identity = meta['controls']['conservative']['identity']
        if (local['controls']['conservative']['identity'] != identity or
            any(local[k] != meta[k] for k in ('rle_sha256','geometry_export_sha256','fixed_moments','pitch','side'))
            or meta['side'] != 256):
            raise ValueError(f'{gid} has a different geometry or material index')
        blocks = BlockMeasure.open(indexes/gid/'conservative',identity=identity,block_voxels=8)
        for path in sorted(operands.glob(gid+'_R*.npz')):
            key=path.stem
            if not PATTERN.fullmatch(key):
                continue
            output_file = output/(key+'.json')
            expected = dict(row_id=key, operand_sha256=digest(path), material_identity=identity,
                            static_material_evidence_sha256=digest(STATIC), bandwidth=meta['pitch'],
                            response_recipe_sha256=identity_sha)
            if output_file.exists():
                existing=read(output_file)
                if not all(existing.get(k)==v for k,v in expected.items()):
                    raise ValueError(f'{key} cached prediction belongs to a different input')
                continue
            operand = FiniteOperand.load(path)
            surface = prepare_surface(operand.vertices, operand.faces)
            with torch.inference_mode():
                result = evaluate_packed(blocks,surface,meta['pitch'],operand.operation,meta['fixed_moments'],
                                         vertex_velocity=operand.vertex_velocity)
            write(output_file,dict(**expected, moments=result.moments.tolist(),
                                   J=result.jacobian.tolist() if result.jacobian is not None else None,
                                   diagnostics=result.diagnostics))
            print(key, result.diagnostics['solver_status'],flush=True)


def summarize(group):
    def metric(field):
        values = [(r[field],r['row_id']) for r in group if r.get(field) is not None]
        if not values:
            return dict(n=0,p95=None,worst=None,worst_row=None)
        worst, key = max(values)
        return dict(n=len(values),p95=float(np.percentile([x[0] for x in values],95)),worst=worst,worst_row=key)
    return dict(denominator=len(group),M=metric('M_scaled_max'),J=metric('J_unit_direction_scaled_max'),
                raw_M_query_rmse=metric('raw_M_rmse'),raw_J_query_rmse=metric('raw_J_rmse'))


def score(archive, predictions, output):
    with gzip.open(archive,'rt') as f:
        data = json.load(f)
    static = read(STATIC)
    settings=recipe()
    identity_sha=recipe_sha(settings)
    inventory=read(HERE/'qualification/evidence/published_inventory.json')
    sources={r['geometry']:r['source_group'] for r in inventory['splits']['development']['records']}
    rows = []
    for row in sorted(data['rows'],key=lambda r:r['row_id']):
        gid, key = row['geometry_id'], row['row_id']
        meta, material = static[gid], row['material']
        if (meta['rle_sha256'] not in row['geometry_files'].values() or
            meta['fixed_moments'] != row['fixed_moments'] or meta['density'] != row['density']):
            raise ValueError(f'{key} changed physical frame or geometry')
        result = dict(row_id=key, geometry_id=gid, source_group=sources[gid],dimension=row['dimension'], operand_sha256=row.get('operand_sha256'),
                      sampled_row_passed=row['sampled_row_passed'], numeric_moment=material['moment_valid'],
                      numeric_J=material.get('restricted_valid',False))
        rows.append(result)
        if not result['operand_sha256']:
            continue
        prediction = read(predictions/(key+'.json'))
        if (prediction['operand_sha256'] != result['operand_sha256'] or
            prediction['material_identity'] != meta['controls']['conservative']['identity'] or
            prediction['static_material_evidence_sha256'] != digest(STATIC) or prediction['bandwidth'] != meta['pitch']
            or prediction['response_recipe_sha256']!=identity_sha):
            raise ValueError(f'{key} prediction uses different source geometry')
        result['prediction'] = prediction
        mass, length = meta['reference_editable_mesh'][0], meta['length']
        if result['numeric_moment']:
            fraction=material['editable_moments'][0]/mass
            result['editable_fraction']=fraction
            result['retention_range']=('near_zero' if fraction<.01 else 'low' if fraction<.1
                                       else 'middle' if fraction<.7 else 'high')
            errors=moment_errors(prediction['moments'],material['moments'],mass,length)
            result['moment_errors']=errors
            result['M_scaled_max']=errors['scaled_max']
            result['raw_M_rmse']=float(np.sqrt(np.mean(np.square(errors['absolute']))))
        if result['numeric_J'] and prediction['J'] is not None:
            errors=jacobian_errors(
                prediction['J'],material['raw_restricted_J'],mass,length,np.linalg.norm(row['query']),
                basis=row['raw_basis'])
            result['jacobian_errors']=errors
            result['J_unit_direction_scaled_max']=errors['unit_direction_scaled_max']
            result['raw_J_rmse']=errors['rmse']
    if len(rows) != 89 or {r['row_id'] for r in rows} != set(historic()[0]):
        raise ValueError('incomplete development cohort')
    grouped = {f'R{dim}':summarize([r for r in rows if r['dimension']==dim]) for dim in (3,8)}
    by_geom = {gid:{f'R{dim}':summarize([r for r in rows if r['geometry_id']==gid and r['dimension']==dim])
                   for dim in (3,8)} for gid in sorted({r['geometry_id'] for r in rows})}
    by_source = {group:{f'R{dim}':summarize([r for r in rows if r['source_group']==group and r['dimension']==dim])
                       for dim in (3,8)} for group in sorted(set(sources.values()))}
    by_fraction = {band:{f'R{dim}':summarize([r for r in rows if r.get('retention_range')==band and r['dimension']==dim])
                         for dim in (3,8)} for band in ('near_zero','low','middle','high')}
    limits=read(HERE.parent/'protocol.json')['screening_limits']
    above={f'R{dim}':{metric:[dict(row_id=r['row_id'],error=r[metric]) for r in rows if r['dimension']==dim
                               and r.get(metric) is not None and r[metric]>limit]
                      for metric,limit in [('M_scaled_max',limits['moment_p95']),
                                           ('J_unit_direction_scaled_max',limits['directional_jacobian_p95_per_full_r3_r8'])]}
           for dim in (3,8)}
    write(output,dict(scope='Frozen deterministic V2.1 conservative finite-operand response, zero learned weights. '
                      'Descriptive numerical errors on all available centres. No event or optimization claim.',
                      physical_replay_sha256=digest(archive), static_material_sha256=digest(STATIC),
                      response_recipe=settings,response_recipe_sha256=identity_sha,
                      by_stratum=grouped,by_geometry=by_geom,by_source_group=by_source,by_retention=by_fraction,
                      individual_queries_above_screen_values=above,rows=rows))


def compare_derived(expected,actual,path):
    if type(expected) is not type(actual):
        raise ValueError(f'{path}: score type changed')
    if isinstance(expected,dict):
        if expected.keys()!=actual.keys():
            raise ValueError(f'{path}: score fields changed')
        for key in expected:
            compare_derived(expected[key],actual[key],f'{path}.{key}')
    elif isinstance(expected,list):
        if len(expected)!=len(actual):
            raise ValueError(f'{path}: score population changed')
        for index,(left,right) in enumerate(zip(expected,actual,strict=True)):
            compare_derived(left,right,f'{path}[{index}]')
    elif isinstance(expected,float):
        if not math.isclose(expected,actual,rel_tol=64*np.finfo(np.float64).eps):
            raise ValueError(f'{path}: rescaled error or statistic changed')
    elif expected!=actual:
        raise ValueError(f'{path}: decision, count or identity changed')


def verify_scores(archive,report):
    recorded=read(report)
    with TemporaryDirectory() as work:
        tmp=Path(work)
        for row in recorded['rows']:
            if 'prediction' in row:
                write(tmp/(row['row_id']+'.json'),row['prediction'])
        score(archive,tmp,tmp/'checked.json')
        checked=read(tmp/'checked.json')
        for field in ('scope','physical_replay_sha256','static_material_sha256','response_recipe','response_recipe_sha256'):
            if recorded[field]!=checked[field]:
                raise ValueError(f'{field}: different scientific input')
        for saved,fresh in zip(recorded['rows'],checked['rows'],strict=True):
            if saved.get('prediction')!=fresh.get('prediction'):
                raise ValueError(f"{saved['row_id']}: different physical conditioning or prediction")
        # QR, dot products and percentiles can differ in their last bits between BLAS/NumPy builds.
        # Only recalculated floating-point errors use a tolerance; the stored vectors and decisions compare exactly.
        for field in ('by_stratum','by_geometry','by_source_group','by_retention','individual_queries_above_screen_values','rows'):
            compare_derived(recorded[field],checked[field],field)
        if recorded.keys()!=checked.keys():
            raise ValueError('score report fields changed')


if __name__=='__main__':
    cli=argparse.ArgumentParser(description=__doc__)
    modes=cli.add_subparsers(dest='mode',required=True)
    pred=modes.add_parser('predict')
    for name in ('operands','indexes','predictions'):
        pred.add_argument(name,type=Path)
    scoring=modes.add_parser('score')
    for name in ('archive','predictions','output'):
        scoring.add_argument(name,type=Path)
    verification=modes.add_parser('verify')
    verification.add_argument('archive',type=Path)
    verification.add_argument('report',type=Path)
    args=cli.parse_args()
    if args.mode=='predict':
        predict(args.operands,args.indexes,args.predictions)
    elif args.mode=='score':
        score(args.archive,args.predictions,args.output)
    else:
        verify_scores(args.archive,args.report)
