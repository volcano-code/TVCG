import importlib,unittest
import tempfile
from pathlib import Path

class Aggregation(unittest.TestCase):
 def setUp(self):
  self.assertTrue(Path(__file__).with_name('aggregate.py').exists(),'conservative aggregation missing')
  self.a=importlib.import_module('aggregate')
 def test_missing_evidence_never_passes(self):
  self.assertEqual(self.a.screen([0.,0.,0.],[0.,0.,0.],False,1e-11),'pending')
 def test_observed_failure_not_hidden_by_pending(self):
  self.assertEqual(self.a.screen([0.,2e-11],[0.],False,1e-11),'fail')
 def test_strict_preregistered_gate(self):
  self.assertEqual(self.a.screen([1e-11],[1e-11],True,1e-11),'pass')
  self.assertEqual(self.a.screen([1.01e-11],[0.],True,1e-11),'fail')
 def test_nonfinite_not_pass(self):
  self.assertEqual(self.a.screen([float('nan')],[0.],True,1e-11),'pending')
 def test_missing_run_receipt_retained_as_pending(self):
  self.assertTrue(hasattr(self.a,'read_optional_run'),'missing-run handling absent')
  with tempfile.TemporaryDirectory() as d:
   receipt,report=self.a.read_optional_run(Path(d))
   self.assertEqual(receipt['status'],'pending');self.assertIsNone(report)
 def test_missing_nominal_task_result_stays_pending(self):
  self.assertTrue(hasattr(self.a,'task_results'),'optional task results missing')
  result=self.a.task_results({},[128,256,384,512])
  self.assertEqual(len(result),8)
  self.assertTrue(all(x['status']=='pending' for x in result))
 def test_last_pair_cannot_bridge_missing_resolution(self):
  self.assertTrue(hasattr(self.a,'grid_series'),'adjacent-pair series missing')
  records={}
  for n in [128,256,512]:
   c=dict(retained_moments=[float(n)]*10,retained_topology=dict(total_voxels=n,components=1),retained_volume=float(n),reference_volume=float(n),signed_normalized_moment_error=[0.]*10,max_normalized_moment_error=0.,normalized_volume_error=0.,expected_components=1,partition_residual_normalized=[0.]*10)
   records[('m83','exact',n,'centre')]=dict(execution_status='pass',cases={'full_retention':c},normalization_scales=[1.]*10)
  data=self.a.grid_series(records,'m83','exact','centre','full_retention',[128,256,384,512])
  self.assertIsNone(data[-1]['successive_normalized_moment_change'])
  self.assertIsNotNone(data[1]['successive_normalized_moment_change'])
if __name__=='__main__':unittest.main()
