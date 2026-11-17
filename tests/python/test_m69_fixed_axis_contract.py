import unittest
import json
from pathlib import Path
import numpy as np
from research.NeuralMoments.m69_fixed_axis_contract_v1 import TASK_HASH,evaluate_m69_moments,physical_metrics

TASK = {
 "schema":"m69.fixed-axis-task.v1","axis":[0.,0.,1.],"axis_origin":[0.,0.,0.],"contact":[0.,0.,0.],
 "density":1.,"shell_fraction":.03,"protect_radius_fraction":.04,"protect_height_fraction":0.,
 "polar_retention":.9,"min_removal_fraction":.1,"max_centroid_rise":1e-7,"com_tolerance":1e-4,
 "alignment_tolerance":1e-4,"principal_gap_min":-1e-8,"require_connected":True,
 "mass_tolerance":1e-10,"inertia_tolerance":1e-10,"moment_psd_tolerance":1e-9,
 "solver_tolerance":1e-9,"solver_polar_margin":1e-5,"training_polar_margin":5e-4,
 "objective":"minimum_retained_mass","material_policy":"single_density_removal_only",
 "inertia_reference":"original_fixed_axis"
}
ORIGINAL=np.array([10.,0.,0.,0.,2.,2.,1.,0.,0.,0.])
CANDIDATE=np.array([9.,0.,0.,0.,1.8,1.8,.9,0.,0.,0.])

class M69FixedAxisContractTests(unittest.TestCase):
 def test_registered_task_fixture_contains_all_28_source_fields(self):
  path=Path("research/NeuralMoments/fixtures/m69_fixed_axis_task_registration_v1.json")
  record=json.loads(path.read_text())
  payload=record["task"]["payload"]
  expected={"alignment_tolerance","axis","axis_origin","com_tolerance","contact","density",
   "inertia_reference","inertia_tolerance","mass_tolerance","material_policy","max_centroid_rise",
   "min_removal_fraction","moment_psd_tolerance","name","native_resolution","objective",
   "polar_retention","principal_gap_min","protect_height_fraction","protect_radius_fraction",
   "purpose","require_connected","schema","shell_fraction","solver_polar_margin",
   "solver_tolerance","training_polar_margin"}
  self.assertEqual(set(payload),expected)
  self.assertEqual(record["task"]["task_hash"],TASK_HASH)
  self.assertEqual(payload["inertia_reference"],"original_fixed_axis")
 def test_all_ten_constraints_match_boundary_fixture(self):
  r=evaluate_m69_moments(CANDIDATE,ORIGINAL,TASK,length_scale=2.,connectivity=True)
  self.assertEqual(r.task_hash,TASK_HASH)
  self.assertTrue(r.moment_pass)
  self.assertTrue(r.complete_pass)
  self.assertEqual(len(r.constraint_values),10)
  self.assertTrue(all(r.checks.values()))
 def test_connectivity_is_not_inferred(self):
  r=evaluate_m69_moments(CANDIDATE,ORIGINAL,TASK,length_scale=2.,connectivity=None)
  self.assertTrue(r.moment_pass)
  self.assertFalse(r.connectivity_checked)
  self.assertIsNone(r.connectivity_pass)
  self.assertFalse(r.complete_pass)
 def test_disconnected_binary_result_fails_complete_gate(self):
  r=evaluate_m69_moments(CANDIDATE,ORIGINAL,TASK,length_scale=2.,connectivity=False)
  self.assertTrue(r.moment_pass)
  self.assertFalse(r.complete_pass)
 def test_minimum_removal_is_distinct_from_polar_retention(self):
  c=CANDIDATE.copy(); c[0]=9.5; c[4:]*=9.5/9.
  r=evaluate_m69_moments(c,ORIGINAL,TASK,length_scale=2.,connectivity=True)
  self.assertFalse(r.checks["minimum_removal"])
  self.assertTrue(r.checks["polar_retention"])
 def test_contact_must_be_on_fixed_axis(self):
  bad=dict(TASK); bad["contact"]=[.1,0.,0.]
  with self.assertRaisesRegex(ValueError,"prescribed fixed axis"):
   physical_metrics(ORIGINAL,bad,2.)
 def test_length_scale_must_be_positive(self):
  with self.assertRaisesRegex(ValueError,"length_scale"):
   evaluate_m69_moments(CANDIDATE,ORIGINAL,TASK,length_scale=0.)
 def test_material_policy_is_frozen(self):
  bad=dict(TASK); bad["material_policy"]="variable_density"
  with self.assertRaisesRegex(ValueError,"material policy"):
   evaluate_m69_moments(CANDIDATE,ORIGINAL,bad,length_scale=2.)
if __name__=="__main__": unittest.main()
