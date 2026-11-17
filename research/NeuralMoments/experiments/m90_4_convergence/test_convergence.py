import importlib,unittest,ctypes
from pathlib import Path
import numpy as np

class Convergence(unittest.TestCase):
 def setUp(self):
  self.assertTrue(Path(__file__).with_name('engine.py').exists(),'independent convergence engine missing')
  self.e=importlib.import_module('engine')
 def test_native_matches_frozen_dense_small_grids(self):
  e=self.e
  for n in (12,19):
   for tetra in (False,True):
    v,f=e.box_mesh([.11,.17,.13],[.91,.83,.93])
    if tetra:
     v=np.array([[.13,.12,.11],[.91,.17,.19],[.21,.89,.23],[.27,.31,.93]])
     f=np.array([[0,2,1],[0,1,3],[1,2,3],[2,0,3]],np.int64)
    origin=np.zeros(3);h=1./n
    sparse,_=e.mesh68_runs(v,f,origin,h,n,[1.73e-7,3.17e-7])
    mask=np.zeros((n,n,n),np.uint8);stats=np.zeros(3,np.int64)
    lib=e.library();fn=lib.mesh68_inset
    fn.argtypes=[ctypes.c_void_p,ctypes.c_void_p,ctypes.c_int,ctypes.c_void_p,ctypes.c_double,ctypes.c_int,ctypes.c_double,ctypes.c_double,ctypes.c_double,ctypes.c_void_p,ctypes.c_void_p]
    status=fn(v.ctypes.data,f.ctypes.data,len(f),origin.ctypes.data,h,n,0.,0.,float('-inf'),mask.ctypes.data,stats.ctypes.data)
    self.assertEqual(status,0)
    got=np.zeros_like(mask)
    for x,y,lo,hi in sparse:got[x,y,lo:hi]=1
    np.testing.assert_array_equal(got,mask)
 def test_membership_not_just_count(self):
  a=np.array([[0,0,0,3],[1,0,1,4]],np.int64);b=np.array([[0,0,1,4],[1,0,1,4]],np.int64)
  m=self.e.membership(a,b,8)
  self.assertEqual(m['xor_voxels'],2);self.assertEqual(m['intersection_voxels'],5)
  self.assertEqual(m['count_a'],m['count_b'])
 def test_budget_derivation_decimal_not_binary_rounding(self):
  import run_case
  self.assertTrue(hasattr(run_case,'verify_budget'),'exact declared-tolerance verification missing')
  self.assertTrue(run_case.verify_budget(1e-10,1e-10,1e-11))
  self.assertFalse(run_case.verify_budget(1e-10,1e-10,1e-10))
 def test_all_ten_prism_moments(self):
  rows=np.array([[0,0,0,3]],np.int64);o=np.array([-.7,.2,1.3]);h=.13;rho=2.7
  expected=self.e.cuboid(o,o+[h,h,3*h])*rho
  np.testing.assert_allclose(self.e.moments(rows,o,h,rho),expected,rtol=1e-12,atol=1e-14)
 def test_whole_retained_topology_controls(self):
  e=self.e;n=12;o=np.zeros(3);h=1.
  outer=e.box_runs([1,1,1],[11,11,11],o,h,n)
  cavity=e.box_runs([4,4,4],[8,8,8],o,h,n)
  slab=e.box_runs([1,1,5],[11,11,7],o,h,n)
  self.assertEqual(e.components(e.subtract(outer,cavity,n),n)['components'],1)
  self.assertEqual(e.components(e.subtract(outer,slab,n),n)['components'],2)
 def test_exact_vertical_boundary_pending(self):
  e=self.e;v,f=e.box_mesh([1.5,1,1],[6.5,7,7])
  with self.assertRaises(e.Pending):e.exact_runs(v,f,np.zeros(3),1.,8,[0.,0.])
 def test_invalid_missing_and_outside_mesh(self):
  e=self.e;v,f=e.box_mesh([0,0,0],[1,1,1])
  with self.assertRaises(e.Pending):e.mesh68_runs(None,None,np.zeros(3),.1,12,[0,0])
  with self.assertRaises(e.Pending):e.mesh68_runs(v+2,f,np.zeros(3),.1,12,[0,0])
  with self.assertRaises(e.Pending):e.mesh68_runs(v,f[:-1],np.zeros(3),.1,12,[0,0])
 def test_phase_changes_sampling_not_prism_lattice(self):
  e=self.e;v,f=e.box_mesh([1.50001,1,1],[6.49,7,7]);o=np.zeros(3)
  a,_=e.mesh68_runs(v,f,o,1.,8,[0,0]);b,_=e.mesh68_runs(v,f,o,1.,8,[1e-4,1e-4])
  self.assertGreater(e.membership(a,b,8)['xor_voxels'],0)
  # A selected voxel always integrates its original [i,i+1] footprint.
  np.testing.assert_allclose(e.moments(np.array([[1,2,3,4]]),o,1.,1.),e.cuboid([1,2,3],[2,3,4]),rtol=0,atol=1e-14)

if __name__=='__main__':unittest.main()
