import unittest
import numpy as np


def encode(a):
    r=[]
    for x,y in np.ndindex(a.shape[:2]):
        d=np.diff(np.r_[False,a[x,y],False].astype(int))
        r.extend([x,y,int(lo),int(hi)] for lo,hi in zip(np.flatnonzero(d==1),np.flatnonzero(d==-1)))
    return np.array(r,dtype=np.int64).reshape(-1,4)

def box_mesh(lo,hi):
    x,y,z=lo;X,Y,Z=hi
    v=np.array([[x,y,z],[X,y,z],[X,Y,z],[x,Y,z],[x,y,Z],[X,y,Z],[X,Y,Z],[x,Y,Z]],float)
    f=np.array([[0,2,1],[0,3,2],[4,5,6],[4,6,7],[0,1,5],[0,5,4],[1,2,6],[1,6,5],[2,3,7],[2,7,6],[3,0,4],[3,4,7]])
    return v,f

class SparseControls(unittest.TestCase):
    def test_face_edge_corner(self):
        from sparse import components
        for other,want in [([0,0,1,2],1),([0,1,0,1],1),([1,0,0,1],1),([1,1,0,1],2),([1,1,1,2],2),([1,0,1,2],2)]:
            rows=np.array([[0,0,0,1],other])
            # The adjacent z cells must be canonicalized by the producer.
            if other==[0,0,1,2]:rows=np.array([[0,0,0,2]])
            with self.subTest(other=other):self.assertEqual(components(rows,4)['components'],want)

    def test_invalid_rle(self):
        from sparse import components
        for r in [np.array([[0,0,0.,1.]]),np.array([[0,0,0,1],[0,0,0,1]]),np.array([[0,0,0,2],[0,0,1,3]]),np.array([[0,0,0,1],[0,0,1,2]]),np.array([[4,0,0,1]]),np.array([[0,0,-1,1]]),np.array([[0,0,2,1]]),np.array([[1,0,0,1],[0,0,0,1]])]:
            with self.subTest(r=r.tolist()):
                with self.assertRaises(ValueError):components(r,4)

    def test_difference_and_support(self):
        from sparse import subtract
        A=np.array([[1,1,0,8],[1,2,0,8]])
        B=np.array([[1,1,2,4],[1,2,1,7]])
        np.testing.assert_array_equal(subtract(A,B,8),[[1,1,0,2],[1,1,4,8],[1,2,0,1],[1,2,7,8]])
        with self.assertRaises(ValueError):subtract(A,np.array([[0,0,0,1]]),8)

    def test_cavity_and_disconnected_retained(self):
        from sparse import subtract,components
        full=np.ones((8,)*3,bool);cut=np.zeros_like(full);cut[2:4,2:4,2:4]=True;cut[5:7,5:7,5:7]=True
        self.assertEqual(components(encode(cut),8)['components'],2)
        self.assertEqual(components(subtract(encode(full),encode(cut),8),8)['components'],1)
        cut[:]=False;cut[4,:,:]=True
        self.assertEqual(components(subtract(encode(full),encode(cut),8),8)['components'],2)

    def test_dense_differential(self):
        from scipy.ndimage import label
        from sparse import components,subtract
        rng=np.random.default_rng(903)
        for p in [.02,.15,.4,.75,1.]:
            a=rng.random((12,)*3)<p;b=a&(rng.random(a.shape)<.3)
            lab,k=label(a&~b);sizes=np.bincount(lab.ravel())[1:]
            got=components(subtract(encode(a),encode(b),12),12)
            self.assertEqual(got['components'],k);self.assertEqual(got['largest_voxels'],max(sizes,default=0));self.assertEqual(got['total_voxels'],int((a&~b).sum()))

    def test_mesh_cube(self):
        from sparse import mesh_runs,components
        v,f=box_mesh([1,1,1],[7,7,7]);r,meta=mesh_runs(v,f,[0,0,0],1.,8)
        expected=np.zeros((8,)*3,bool);expected[1:7,1:7,1:7]=True
        np.testing.assert_array_equal(r,encode(expected));self.assertEqual(components(r,8)['total_voxels'],216)
        self.assertFalse(meta['dense_n_cubed_allocated'])

    def test_mesh_pending_cases(self):
        from sparse import mesh_runs,Pending
        v,f=box_mesh([1,1,1],[7,7,7])
        with self.assertRaises(Pending):mesh_runs(v,f[:-1],[0,0,0],1.,8)
        with self.assertRaises(Pending):mesh_runs(v,f,[0,0,0],1.,8,max_projected_tests=1)
        with self.assertRaises(Pending):mesh_runs(v+.5,f,[0,0,0],1.,8)
        with self.assertRaises(Pending):mesh_runs(v,f,[2,2,2],1.,8)
        with self.assertRaises(Pending):mesh_runs(None,None,[0,0,0],1.,8)

    def test_translated_scaled(self):
        from sparse import mesh_runs
        v,f=box_mesh([1,1,1],[7,7,7]);r,_=mesh_runs(v,f,[0,0,0],1.,8)
        s,_=mesh_runs(v*2+[-5,2,3],f,[-5,2,3],2.,8)
        np.testing.assert_array_equal(r,s)

    def test_empty(self):
        from sparse import components,subtract
        e=np.empty((0,4),int);self.assertEqual(components(e,8)['components'],0);self.assertEqual(subtract(e,e,8).shape,(0,4))


    def test_vertical_surface_centres_pending(self):
        from sparse import mesh_runs,Pending
        for lo,hi in [([1.5,1,1],[6.5,7,7]),([1,1.5,1],[7,6.5,7])]:
            v,f=box_mesh(lo,hi)
            with self.subTest(lo=lo):
                with self.assertRaises(Pending):mesh_runs(v,f,[0,0,0],1.,8)


    def test_guarded_legacy_positive_and_ambiguous(self):
        import sys,tempfile,json
        from pathlib import Path
        sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'m90_2_saved_result_validator'))
        from test_validator import fixture,save,sha
        from guarded_validator import validate
        for ambiguous in [False,True]:
            with tempfile.TemporaryDirectory() as td:
                p=Path(td);g,r=fixture(p)
                if ambiguous:
                    with np.load(p/'outer.npz') as a:v=a['vertices'];f=a['faces']
                    v[:,0]=np.where(v[:,0]>0,2.5,-2.5)
                    np.savez(p/'outer.npz',vertices=v,faces=f);g['outer_mesh']['sha256']=sha(p/'outer.npz')
                a,b=save(p,g,r);report=validate(p/'registry.json',p/'result.json',registry_sha256=a,result_sha256=b)
                self.assertEqual(report['status'],'pending' if ambiguous else 'pass')
                if ambiguous:self.assertIsNone(report['claim'])

if __name__=='__main__':unittest.main()


