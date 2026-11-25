"""Two actual default-model CPU updates, with independent interrupted replay.

All assets are explicit analytic fixtures. This checks software transport and
exact continuation, not neural operator quality or native solver acceptance.
"""
from __future__ import annotations

import argparse
import copy
import gc
import json
from pathlib import Path

import numpy as np
import torch

from ..p0_diagnostics.baseline_cases import (
    LABEL_IDENTITY_FIELDS, operand_fingerprint, query_fingerprint, sha256, write_synthetic_finite_case,
)
from ..p0_diagnostics.exposure_audit import read_events, summarize_events
from .manifest import SCHEMA, load_training_manifest
from .runner import Trainer, TrainConfig, _json_write


def make_smoke_manifest(directory):
    """Two distinct plane queries on the same training-only analytic geometry."""
    from research.NeuralMoments.models.pcraft_ufno.candidate_projects.quadrann_reproduction.spin_adapter.v21_native import FiniteOperand
    directory=Path(directory)
    directory.mkdir(parents=True,exist_ok=False)
    first=write_synthetic_finite_case(directory/'case0')
    spec=json.loads(first.read_text())
    second=directory/'case1'; second.mkdir()
    copied=copy.deepcopy(spec)
    for name in ('geometry','material'):
        copied['assets'][name]['path']=str((first.parent/spec['assets'][name]['path']).resolve())
    operand=FiniteOperand.load(first.parent/'operand.npz')
    vertices=operand.vertices.copy()
    vertices[vertices[:,0] == .037,0]=.049
    moved=FiniteOperand(vertices,operand.faces,operand.face_is_cap,operand.operation,operand.vertex_velocity)
    moved.save(second/'operand.npz')
    coefficients=np.r_[-.049,1.,np.zeros(8)]
    meta=copied['meta']; meta['query_id']='finite_plane_0049'
    meta['operand_content_sha256']=operand_fingerprint(moved)
    meta['query_content_sha256']=query_fingerprint(coefficients)
    with np.load(first.parent/'targets.npz',allow_pickle=False) as data:
        targets={key:data[key].copy() for key in data.files}
    lo,hi=np.array([-.21,-.17,-.13]),np.array([.049,.23,.18])
    center=(lo+hi)/2; extent=hi-lo; x,y,z=center
    values=np.array([1.,x,y,z,x*x,y*y,z*z,y*z,x*z,x*y]); values[4:7]+=extent**2/12
    targets['native_moments']=targets['fixed_moments']+1.3*extent.prod()*values
    def integral(axis,power):
        return (hi[axis]**(power+1)-lo[axis]**(power+1))/(power+1)
    powers=[(0,0,0),(1,0,0),(0,1,0),(0,0,1),(2,0,0),(0,2,0),(0,0,2),(0,1,1),(1,0,1),(1,1,0)]
    targets['native_jacobian']=np.array([[.049**px*1.3*integral(1,py+dy)*integral(2,pz+dz)
        for dy,dz in ((0,0),(1,0),(0,1))] for px,py,pz in powers])
    targets['coefficients']=coefficients
    targets['metadata']=np.asarray(json.dumps({key:meta[key] for key in LABEL_IDENTITY_FIELDS}))
    np.savez(second/'targets.npz',**targets)
    for name in ('operand','targets'):
        copied['assets'][name]=dict(path=name+'.npz',sha256=sha256(second/(name+'.npz')))
    second_case=second/'case.json'; second_case.write_text(json.dumps(copied,indent=2)+'\n')
    records=[dict(record_id=f'fixture-{i}',case_path=str(path.relative_to(directory)),case_sha256=sha256(path),
                  role='train',loader_eligible=True,moment_weight=1.,jacobian_weight=1.,
                  surface_family='PLANE',retention_bin='middle',trajectory_stage='synthetic_fixture')
             for i,path in enumerate((first,second_case))]
    manifest=directory/'training_manifest.json'
    manifest.write_text(json.dumps(dict(schema=SCHEMA,snapshot_id='synthetic-real-operator-software-smoke-v1',
                                        registered_train_count=2,records=records),indent=2)+'\n')
    return manifest


def _exact(left,right):
    if isinstance(left,torch.Tensor):
        return isinstance(right,torch.Tensor) and torch.equal(left,right)
    if type(left) is not type(right):
        return False
    if isinstance(left,dict):
        return left.keys()==right.keys() and all(_exact(left[key],right[key]) for key in left)
    if isinstance(left,(tuple,list)):
        return len(left)==len(right) and all(_exact(a,b) for a,b in zip(left,right))
    return left==right


def run_smoke(output):
    output=Path(output).resolve(); output.mkdir(parents=True,exist_ok=False)
    manifest=make_smoke_manifest(output/'assets')
    data=load_training_manifest(manifest,mode='smoke',cache_directory=output/'cache')
    config=TrainConfig()
    # First run is the uninterrupted reference. Its initial checkpoint is also
    # the common starting point of the interrupted branch.
    trainer=Trainer(data,output/'continuous',config=config,max_steps=2,max_attempts=4,run_id='synthetic-exact-resume')
    initial=trainer.save_checkpoint()
    first_row=trainer.step()
    middle=trainer.save_checkpoint()
    second_row=trainer.step()
    continuous=trainer.save_checkpoint()
    trainer.close(); del trainer; gc.collect()
    # The middle checkpoint was produced after the first real update. Restore
    # it into a new process segment and independently replay the second update.
    resumed=Trainer(data,output/'resumed',config=config,max_steps=2,max_attempts=4,
                    resume=middle['receipt_path'])
    replayed=resumed.step()
    replay_checkpoint=resumed.save_checkpoint()
    resumed.close(); del resumed; gc.collect()
    expected=torch.load(continuous['path'],map_location='cpu',weights_only=True)
    observed=torch.load(replay_checkpoint['path'],map_location='cpu',weights_only=True)
    keys=('model','optimizer','scheduler','scaler','rng','sampler','global_step','attempts')
    checks={key:bool(_exact(expected[key],observed[key])) for key in keys}
    checks['same_second_batch']=second_row['record_ids']==replayed['record_ids']
    checks['same_second_loss']=second_row['loss']==replayed['loss']
    checks['parameters_updated']=initial['model_tensor_sha256'] != continuous['model_tensor_sha256']
    checks['joint_loss_effective']=all(row['jacobian_denominator']>0 and row['moment_denominator']>0
                                       for row in (first_row,second_row,replayed))
    checks['encoder_and_response_finite_nonzero_gradients']=all(
        group['finite'] and group['norm'] is not None and group['norm']>0
        for row in (first_row,second_row,replayed) for group in row['gradients'].values())
    events=read_events([output/'continuous'/'exposure.jsonl',output/'resumed'/'exposure.jsonl'])
    selected=json.loads(Path(replay_checkpoint['receipt_path']).read_text())['exposure_receipt']['checkpoint_id']
    exposure=summarize_events(events,selected_checkpoint=selected)
    report=dict(schema='compact_pino.actual-operator-smoke.v1',scope='synthetic_software_only',
                status='PASS' if all(checks.values()) else 'FAIL',checks=checks,
                config=expected['config'],training_manifest_path=str(manifest),
                training_manifest_sha256=data.manifest_sha256,initial_checkpoint=initial,
                interrupted_boundary_checkpoint=middle,continuous_checkpoint=continuous,
                resumed_checkpoint=replay_checkpoint,steps=[first_row,second_row,replayed],
                exposure=exposure,real_optimizer_calls_executed=3,
                native_solver_acceptance='NOT_EVALUATED',scientific_promotion='NOT_EVALUATED',
                exactness_scope='same CPU runtime, deterministic algorithms; GPU exact continuation not validated here')
    _json_write(output/'smoke_report.json',report)
    if not all(checks.values()):
        raise AssertionError('actual operator smoke failed: '+str(checks))
    return report


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',required=True)
    args=parser.parse_args()
    result=run_smoke(args.output)
    print(json.dumps(dict(status=result['status'],scope=result['scope'],checks=result['checks'],
                         report=str(Path(args.output)/'smoke_report.json')),indent=2))


if __name__=='__main__': main()
