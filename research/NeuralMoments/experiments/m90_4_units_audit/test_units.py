"""Read-only characterization of frozen M69 semantics; no solver or training."""
import sys, unittest
from pathlib import Path
from dataclasses import replace
from hashlib import sha256
import numpy as np

SRC=Path('/data/xvxinyao/Spin_M68_Trellis_Data_Kit/src')
EXPECTED={'task69.py':'5463dcfe57f3aea21f577fd77d26867ef9e7ec18098b7ff25e7bde5209fc3389',
          'oracle69.py':'8ab8eec248e5497e7d9864d4b91cb18416bd72ab95623eeab4c4f98c8b35dd01'}
for name,digest in EXPECTED.items():
 if sha256((SRC/name).read_bytes()).hexdigest()!=digest:raise RuntimeError('frozen source mismatch: '+name)
sys.path.insert(0,str(SRC))
from task69 import TaskSpec,evaluate_moments,transform_moments
from oracle69 import NativeQuadricOracle

def box():
 # Independent uniform box [-1,1] x [-1,1] x [0,.4], rho=1.
 return np.array([1.6,0,0,.32,1.6/3,1.6/3,1.6*.16/3,0,0,0])

class Units(unittest.TestCase):
 def test_all_ten_constraints_scale_and_density_covariance(self):
  original=box();remaining=.8*original;task=TaskSpec()
  base=evaluate_moments(remaining,original,task,2.)
  powers=np.array([3,4,4,4,5,5,5,5,5,5])
  for scale in (1e-3,7.,1e3):
   factor=2.3*scale**powers
   got=evaluate_moments(remaining*factor,original*factor,replace(task,density=2.3),2.*scale)
   self.assertEqual(len(got['constraint_values']),10)
   np.testing.assert_allclose(list(got['constraint_values'].values()),list(base['constraint_values'].values()),rtol=2e-13,atol=2e-14)
   self.assertIsNone(got['connectivity_pass'])
 def test_com_and_height_tolerances_are_length_fractions(self):
  task=TaskSpec();original=box();L=2.
  delta=np.array([task.com_tolerance*L*1.5,0,task.max_centroid_rise*L*1.5])
  moved=transform_moments(original,translation=delta)
  r=evaluate_moments(moved,original,task,L)
  self.assertAlmostEqual(r['constraint_values']['axis_alignment_com'],-.5*task.com_tolerance,places=14)
  self.assertAlmostEqual(r['constraint_values']['centroid_height'],-.5*task.max_centroid_rise,places=14)
  self.assertFalse(r['checks']['axis_alignment_com'])
  self.assertFalse(r['checks']['centroid_height'])
 def test_oracle_default_grid_span_differs_from_outer_length(self):
  o=NativeQuadricOracle(np.array([[0,0,0,1]],dtype=np.int64),[0,0,0],1.02/512,512)
  self.assertAlmostEqual(o.length_scale,1.02)
  task=TaskSpec();original=box()
  moved=transform_moments(original,translation=[task.com_tolerance*1.01,0,0])
  self.assertFalse(evaluate_moments(moved,original,task,1.)['checks']['axis_alignment_com'])
  self.assertTrue(evaluate_moments(moved,original,task,o.length_scale)['checks']['axis_alignment_com'])
 def test_contact_axis_absolute_floor_not_scale_invariant(self):
  TaskSpec(contact=(5e-11,0,0))
  with self.assertRaisesRegex(ValueError,'contact must lie'):
   TaskSpec(contact=(5e-8,0,0))

if __name__=='__main__':unittest.main()
