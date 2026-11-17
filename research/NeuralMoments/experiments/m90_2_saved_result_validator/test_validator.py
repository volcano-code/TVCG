"""Behavioral controls: literal cuboid references are independent of validator math."""
import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
import numpy as np

HERE = Path(__file__).parent

def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def cuboid(lo, hi):
    lo,hi=np.array(lo,float),np.array(hi,float)
    m=float(np.prod(hi-lo)); c=(lo+hi)/2; s=(lo*lo+lo*hi+hi*hi)/3
    return np.array([m,*(m*c),*(m*s),m*c[1]*c[2],m*c[0]*c[2],m*c[0]*c[1]])

def encode(mask):
    rows=[]
    for x,y in np.ndindex(mask.shape[:2]):
        d=np.diff(np.r_[False,mask[x,y],False].astype(int))
        rows.extend([x,y,int(a),int(b)] for a,b in zip(np.where(d==1)[0],np.where(d==-1)[0]))
    return rows

def mesh_box(lo,hi):
    a,b,c=lo;d,e,f=hi
    v=np.array([[a,b,c],[d,b,c],[d,e,c],[a,e,c],[a,b,f],[d,b,f],[d,e,f],[a,e,f]],float)
    faces=np.array([[0,2,1],[0,3,2],[4,5,6],[4,6,7],[0,1,5],[0,5,4],[1,2,6],[1,6,5],[2,3,7],[2,7,6],[3,0,4],[3,4,7]])
    return v,faces

def fixture(root, mode='cavity'):
    import sys
    sys.path.insert(0,'/data/xvxinyao/Spin_M68_Trellis_Data_Kit/src')
    from task69 import TaskSpec
    root=Path(root); root.mkdir(parents=True,exist_ok=True)
    occ=np.zeros((8,8,8),bool);occ[1:7,1:7,2:6]=True
    edit=np.zeros_like(occ);edit[2:6,2:6,3:5]=True
    cut=np.zeros_like(occ)
    if mode=='cavity':cut[2:6,3:5,3:5]=True
    original=cuboid([-3,-3,-2],[3,3,2]);editable=cuboid([-2,-2,-1],[2,2,1])
    removed=cuboid([-2,-1,-1],[2,1,1]) if mode=='cavity' else np.zeros(10)
    task=TaskSpec(purpose='engineering',native_resolution=8,min_removal_fraction=0 if mode=='intact' else .1)
    v,f=mesh_box([-3,-3,-2],[3,3,2]);np.savez(root/'outer.npz',vertices=v,faces=f)
    np.savez(root/'original.npz',occupied=occ,editable=edit,protected=np.zeros_like(occ),density=occ.astype(float),original_moments=original,fixed_moments=original-editable)
    registry={'schema':'m90.2.registry.v1','coordinate_space':'registered-task','id':'analytic-flat-box-v1','grid':{'side':8,'origin':[-4.,-4.,-4.],'h':1.},'frame':{'name':'task','world_to_task_rotation':np.eye(3).tolist(),'translation':[0.,0.,0.]},'units':{'length':'m','mass':'kg','density':'kg/m^3'},'backend':'exact-voxel-prisms.v1','length_scale':6.,'density':1.,'task':task.to_dict(),'task_hash':task.task_hash,'task_source_sha256':sha('/data/xvxinyao/Spin_M68_Trellis_Data_Kit/src/task69.py'),'original':{'path':'original.npz','sha256':sha(root/'original.npz')},'outer_mesh':{'path':'outer.npz','sha256':sha(root/'outer.npz')},'generator':{'kind':'analytic-fixture','model_checkpoint':None},'moment_tolerance':1e-10}
    result={'schema':'m90.2.saved-result.v1','id':registry['id'],'role':'deleted','encoding':'canonical-integer-rle.v1','grid':registry['grid'],'frame':registry['frame'],'units':registry['units'],'backend':registry['backend'],'task_hash':task.task_hash,'original_sha256':registry['original']['sha256'],'density':1.,'runs':encode(cut),'removed_moments':removed.tolist(),'retained_moments':(original-removed).tolist(),'fixed_moments':(original-editable).tolist(),'export_provenance':{'source_role':'deleted','operation':'direct-deletion'}}
    return registry,result

def save(root,reg,res):
    root=Path(root)
    (root/'registry.json').write_text(json.dumps(reg,sort_keys=True))
    res=copy.deepcopy(res);res['registry_sha256']=sha(root/'registry.json')
    (root/'result.json').write_text(json.dumps(res,sort_keys=True))
    return sha(root/'registry.json'),sha(root/'result.json')

class ValidatorControls(unittest.TestCase):
    def run_case(self, mode='cavity', mutate=None):
        import validator
        with tempfile.TemporaryDirectory() as td:
            reg,res=fixture(td,mode)
            if mutate:mutate(Path(td),reg,res)
            a,b=save(td,reg,res)
            return validator.validate(Path(td)/'registry.json',Path(td)/'result.json',registry_sha256=a,result_sha256=b)

    def test_validator_exists(self):
        self.assertTrue((HERE/'validator.py').is_file(),'independent saved-result validator is not implemented')

    def test_connected_internal_cavity(self):
        r=self.run_case();self.assertEqual(r['status'],'pass',r);self.assertEqual(len(r['task']['checks']),10)
        self.assertEqual(r['topology']['retained_components'],1);self.assertEqual(r['moments']['removed'][0],16.)
        np.testing.assert_allclose(r['moments']['original'],[144,0,0,0,432,432,192,0,0,0],atol=1e-12)

    def test_intact_homogeneous_solid(self):self.assertEqual(self.run_case('intact')['status'],'pass')
    def test_full_retention_fails_minimum_removal(self):
        r=self.run_case('full');self.assertEqual(r['status'],'fail');self.assertFalse(r['task']['checks']['minimum_removal']);self.assertEqual(r['gates']['A1']['status'],'pass')

    def test_negative_controls(self):
        def density(p,g,r):
            with np.load(p/'original.npz') as a:d={k:a[k] for k in a.files}
            d['density'][2,2,3]=2.;np.savez(p/'original.npz',**d);g['original']['sha256']=sha(p/'original.npz');r['original_sha256']=g['original']['sha256']
        cases={
          'multi_density':(density,'A1'),
          'density_increase':(lambda p,g,r:r.update(density=1.1),'A1'),
          'fractional':(lambda p,g,r:r['runs'].__setitem__(0,[2,3,3.5,5]),'A2'),
          'nonbinary':(lambda p,g,r:r.update(final_density=[.5]),'A2'),
          'fixed_shell':(lambda p,g,r:r.update(runs=[[1,1,2,3]]),'A2'),
          'outside':(lambda p,g,r:r.update(runs=[[0,0,0,1]]),'A2'),
          'swapped_role':(lambda p,g,r:r.update(role='retained_editable'),'A2'),
          'duplicate':(lambda p,g,r:r['runs'].insert(0,r['runs'][0]),'A2'),
          'overlap':(lambda p,g,r:r.update(runs=[[2,3,3,5],[2,3,4,5]]),'A2'),
          'adjacent':(lambda p,g,r:r.update(runs=[[2,3,3,4],[2,3,4,5]]),'A2'),
          'out_of_range':(lambda p,g,r:r.update(runs=[[8,3,3,5]]),'A2'),
          'negative_index':(lambda p,g,r:r.update(runs=[[-1,3,3,5]]),'A2'),
          'wrong_grid':(lambda p,g,r:r.update(grid={**g['grid'],'h':2.}),'A0'),
          'wrong_hash':(lambda p,g,r:r.update(original_sha256='0'*64),'A0'),
          'double_shell':(lambda p,g,r:r['retained_moments'].__setitem__(0,240.),'A3'),
        }
        for name,(mutate,gate) in cases.items():
            with self.subTest(name=name):
                report=self.run_case(mutate=mutate);self.assertEqual(report['status'],'fail',report);self.assertEqual(report['gates'][gate]['status'],'fail',report)

    def test_protected_removal(self):
        def change(p,g,r):
            with np.load(p/'original.npz') as a:d={k:a[k] for k in a.files}
            d['protected'][2,3,3]=True;np.savez(p/'original.npz',**d);g['original']['sha256']=sha(p/'original.npz');r['original_sha256']=g['original']['sha256']
        r=self.run_case(mutate=change);self.assertEqual(r['gates']['A2']['status'],'fail')

    def test_missing_outer_mesh(self):
        r=self.run_case(mutate=lambda p,g,r:g.pop('outer_mesh'));self.assertEqual(r['status'],'pending');self.assertEqual(r['gates']['A5']['status'],'pending')

    def test_missing_units(self):
        r=self.run_case(mutate=lambda p,g,r:g.pop('units'));self.assertEqual(r['status'],'pending')

    def test_impossible_task(self):
        def change(p,g,r):
            from task69 import TaskSpec
            g['task']['polar_retention']=1.;t=TaskSpec.from_dict(g['task']);g['task_hash']=r['task_hash']=t.task_hash
        r=self.run_case(mutate=change);self.assertEqual(r['status'],'fail');self.assertEqual(r['task']['analytic_status']['status'],'analytic_infeasible')

    def test_disconnected_retained(self):
        def change(p,g,r):
            with np.load(p/'original.npz') as a:d={k:a[k] for k in a.files}
            d['editable']=d['occupied'].copy();d['fixed_moments']=np.zeros(10);np.savez(p/'original.npz',**d)
            g['original']['sha256']=sha(p/'original.npz');r['original_sha256']=g['original']['sha256']
            cut=np.zeros((8,8,8),bool);cut[4,1:7,2:6]=True;r['runs']=encode(cut)
            removed=cuboid([0,-3,-2],[1,3,2]);r['removed_moments']=removed.tolist();r['retained_moments']=(d['original_moments']-removed).tolist();r['fixed_moments']=[0.]*10
        r=self.run_case(mutate=change);self.assertEqual(r['gates']['A5']['status'],'fail');self.assertEqual(r['topology']['retained_components'],2)

    def test_explicit_complement(self):
        import validator
        E=[[1,1,0,8],[1,2,0,8]];K=[[1,1,0,2],[1,1,4,8],[1,2,1,7]]
        self.assertEqual(validator.complement_retained(E,K,8),[[1,1,2,4],[1,2,0,1],[1,2,7,8]])
        self.assertEqual(validator.complement_retained(E,E,8),[])
        with self.assertRaises(ValueError):validator.complement_retained(E,[[0,0,0,1]],8)

    def test_tampered_saved_bytes(self):
        import validator
        with tempfile.TemporaryDirectory() as td:
            g,r=fixture(td);a,b=save(td,g,r);p=Path(td)/'result.json';p.write_text(p.read_text()+' ')
            report=validator.validate(Path(td)/'registry.json',p,registry_sha256=a,result_sha256=b)
            self.assertEqual(report['gates']['A0']['status'],'fail')


class IntegrationControls(unittest.TestCase):
    run_case = ValidatorControls.run_case
    # Only added tests, inherited controls are collected once by explicit runner.
    def test_actual_oracle_export_complement(self):
        import validator
        import sys
        sys.path.insert(0,"/data/xvxinyao/Spin_M68_Trellis_Data_Kit/src")
        from oracle69 import NativeQuadricOracle
        with tempfile.TemporaryDirectory() as td:
            g,r=fixture(td)
            with np.load(Path(td)/'original.npz') as d:
                columns=encode(d['editable']);original=d['original_moments'];fixed=d['fixed_moments']
            oracle=NativeQuadricOracle(np.asarray(columns,np.int64),[-4,-4,-4],1.,8,fixed_moments=fixed,original_moments=original)
            # Keep all editable cells: retained export is nonempty, deletion must be empty.
            q=np.array([1.,0,0,0,0,0,0,0,0,0]);kept=oracle.export_rle(q).tolist()
            self.assertTrue(kept);self.assertEqual(validator.complement_retained(columns,kept,8),[])
            np.testing.assert_array_equal(oracle.deleted_rle(q),np.empty((0,4),dtype=np.uint32))
            p=Path(td)/'retained.json';p.write_text(json.dumps({'role':'retained_editable','runs':kept}))
            r.update(runs=[],removed_moments=[0.]*10,retained_moments=original.tolist(),export_provenance={'source_role':'retained_editable','operation':'editable-minus-retained','retained_file':{'path':'retained.json','sha256':sha(p)}})
            a,b=save(td,g,r);report=validator.validate(Path(td)/'registry.json',Path(td)/'result.json',registry_sha256=a,result_sha256=b)
            self.assertEqual(report['gates']['A2']['status'],'pass');self.assertEqual(report['status'],'fail');self.assertFalse(report['task']['checks']['minimum_removal'])
            r['runs']=kept;a,b=save(td,g,r);report=validator.validate(Path(td)/'registry.json',Path(td)/'result.json',registry_sha256=a,result_sha256=b)
            self.assertEqual(report['gates']['A2']['status'],'fail')

    def test_changed_original_reference(self):
        def change(p,g,r):
            with np.load(p/'original.npz') as a:d={k:a[k] for k in a.files}
            d['original_moments'][4]+=1.;np.savez(p/'original.npz',**d)
            g['original']['sha256']=sha(p/'original.npz');r['original_sha256']=g['original']['sha256']
        self.assertEqual(self.run_case(mutate=change)['gates']['A3']['status'],'fail')

    def test_missing_task_field(self):
        self.assertEqual(self.run_case(mutate=lambda p,g,r:g['task'].pop('polar_retention'))['status'],'pending')

    def test_full512_budget(self):
        from geometry import check_budget,Pending
        with self.assertRaises(Pending):check_budget(512)

    def test_translated_scaled_registered_grid(self):
        def change(p,g,r):
            from task69 import TaskSpec
            k=2.;t=np.array([1.,2.,3.]);g['grid']['origin']=(np.array(g['grid']['origin'])*k+t).tolist();g['grid']['h']=k;g['length_scale']*=k
            g['frame']['translation']=t.tolist()
            g['task']['axis_origin']=t.tolist();g['task']['contact']=t.tolist();task=TaskSpec.from_dict(g['task']);g['task_hash']=r['task_hash']=task.task_hash
            with np.load(p/'outer.npz') as a:v=a['vertices']*k+t;f=a['faces']
            np.savez(p/'outer.npz',vertices=v,faces=f);g['outer_mesh']['sha256']=sha(p/'outer.npz')
            with np.load(p/'original.npz') as a:d={j:a[j] for j in a.files}
            original=cuboid(np.array([-3,-3,-2])*k+t,np.array([3,3,2])*k+t);editable=cuboid(np.array([-2,-2,-1])*k+t,np.array([2,2,1])*k+t);removed=cuboid(np.array([-2,-1,-1])*k+t,np.array([2,1,1])*k+t)
            d['original_moments']=original;d['fixed_moments']=original-editable;np.savez(p/'original.npz',**d);g['original']['sha256']=sha(p/'original.npz');r['original_sha256']=g['original']['sha256']
            r.update(removed_moments=removed.tolist(),retained_moments=(original-removed).tolist(),fixed_moments=(original-editable).tolist())
        self.assertEqual(self.run_case(mutate=change)['status'],'pass')

    def test_invented_length_cannot_hide_bad_moments(self):
        def change(p,g,r):
            with np.load(p/'original.npz') as a:d={k:a[k] for k in a.files}
            d['original_moments'][4]+=100.;np.savez(p/'original.npz',**d)
            g['original']['sha256']=sha(p/'original.npz');r['original_sha256']=g['original']['sha256']
            r['retained_moments'][4]+=100.;g['length_scale']=1e8
        self.assertEqual(self.run_case(mutate=change)['status'],'fail')

    def test_missing_coordinate_space_is_pending(self):
        self.assertEqual(self.run_case(mutate=lambda p,g,r:g.pop('coordinate_space',None))['status'],'pending')

    def test_unregistered_frame_is_pending(self):
        self.assertEqual(self.run_case(mutate=lambda p,g,r:g.update(coordinate_space='world-needs-transform'))['status'],'pending')


    def test_actual_saved_nonempty_oracle_complement(self):
        def change(p,g,r):
            from oracle69 import NativeQuadricOracle
            import validator
            with np.load(p/'original.npz') as a:columns=encode(a['editable']);original=a['original_moments'];fixed=a['fixed_moments']
            oracle=NativeQuadricOracle(np.asarray(columns,np.int64),g['grid']['origin'],g['grid']['h'],8,fixed_moments=fixed,original_moments=original)
            q=np.array([-1.,0,0,0,0,1,0,0,0,0]);kept=oracle.export_rle(q).tolist()
            removed=validator.complement_retained(columns,kept,8)
            self.assertEqual(removed,r['runs']);self.assertEqual(removed,oracle.deleted_rle(q).tolist())
            f=p/'retained.json';f.write_text(json.dumps({'role':'retained_editable','runs':kept}))
            r['export_provenance']={'source_role':'retained_editable','operation':'editable-minus-retained','retained_file':{'path':'retained.json','sha256':sha(f)}}
        self.assertEqual(self.run_case(mutate=change)['status'],'pass')

    def test_caller_connectivity_true_does_not_certify(self):
        def change(p,g,r):g.pop('outer_mesh');r['connectivity']=True;r['complete_pass']=True
        self.assertEqual(self.run_case(mutate=change)['status'],'pending')

    def test_partial_task_cannot_use_defaults(self):
        self.assertEqual(self.run_case(mutate=lambda p,g,r:g.update(task={}))['status'],'pending')

    def test_outer_mesh_hash_mismatch(self):
        self.assertEqual(self.run_case(mutate=lambda p,g,r:g['outer_mesh'].update(sha256='0'*64))['gates']['A5']['status'],'fail')

if __name__=='__main__':unittest.main()

