import copy, unittest
from research.NeuralMoments.r3_double_replay_audit_v1 import audit_double_replay

def evidence():
    raw=[2.0,0.5,0.5,0.5,0.2,0.2,0.2,0.01,0.01,0.01]
    order=["M","LX","LY","LZ","QXX","QYY","QZZ","QYZ","QXZ","QXY"]
    meta={"case_id":"case","geometry_hash":"geom","coordinate_frame":"world",
      "length_unit":"m","meters_per_unit":1.0,"density":1000.0,"moment_order":order}
    def replay(run_id):
        ledger={"case_id":"case","geometry_hash":"geom","task_hash":"task",
          "evaluator_identity":"eval","solver_source_commit":"commit","initialization_record_hash":"init",
          "equation_profile":"dual_monomial_raw_v1","moment_metadata":meta,
          "physical_admission":{"authorized":True},"result_acceptance":{"physically_accepted":True},
          "run_receipt":{"run_id":run_id}}
        native={"status":"CONVERGED","event_hash":"event","trace_hash":"trace","input_hash":"input",
          "output_hash":"output","function_evaluations":4,"restart_count":1,
          "restart_function_evaluations":[4],"quadric_coefficients":[1.0]+[0.0]*9,
          "final_total_raw_moments":raw,"final_geometry_hash":"final-geom","final_geometry_verified":True}
        return {"schema":"tvcg.r3.native-replay-bundle.v1","ledger":ledger,
          "native_result":native,"ledger_validation":{"valid":True}}
    def independent():
        return {"case_id":"case","task_hash":"task","source_geometry_hash":"geom",
          "backend_identity":"independent_tetra_exact_v1","backend_sha256":"backend-sha",
          "raw_moments":raw.copy(),"moment_order":order}
    def physical():
        return {"case_id":"case","task_hash":"task","geometry_hash":"geom",
          "overall_pass":True,"hard_constraints_pass":True,"tolerance_source_reference":"test fixture"}
    def geometry():
        return {"case_id":"case","task_hash":"task","geometry_hash":"geom",
          "geometry_integrity_pass":True,"native_independent_equivalence":{"reviewed_equivalent":True}}
    return {"replay_a":replay("run-a"),"replay_b":replay("run-b"),
      "independent_a":independent(),"independent_b":independent(),
      "physical_a":physical(),"physical_b":physical(),"geometry_a":geometry(),"geometry_b":geometry()}

class R3DoubleReplayAuditTests(unittest.TestCase):
    def test_valid_double_replay(self):
        r=audit_double_replay(evidence())
        self.assertTrue(r["double_replay_pass"])
        self.assertTrue(r["native_replay_scientific_match"])
        self.assertTrue(r["distinct_native_execution_ids"])
    def test_same_run_id_rejected(self):
        x=evidence();x["replay_b"]["ledger"]["run_receipt"]["run_id"]="run-a"
        with self.assertRaisesRegex(ValueError,"distinct"):audit_double_replay(x)
    def test_changed_native_event_hash_fails(self):
        x=evidence();x["replay_b"]["native_result"]["event_hash"]="different"
        r=audit_double_replay(x)
        self.assertFalse(r["native_replay_scientific_match"])
        self.assertFalse(r["double_replay_pass"])
    def test_physical_failure_rejected(self):
        x=evidence();x["physical_b"]["overall_pass"]=False
        with self.assertRaisesRegex(ValueError,"physical verification b failed"):audit_double_replay(x)
    def test_geometry_failure_rejected(self):
        x=evidence();x["geometry_a"]["geometry_integrity_pass"]=False
        with self.assertRaisesRegex(ValueError,"geometry integrity a failed"):audit_double_replay(x)
    def test_independent_moment_difference_rejected(self):
        x=evidence();x["independent_b"]["raw_moments"][9]+=1e-3
        with self.assertRaisesRegex(ValueError,"moment comparison b failed"):audit_double_replay(x)
    def test_task_mismatch_rejected(self):
        x=evidence();x["independent_a"]["task_hash"]="other"
        with self.assertRaisesRegex(ValueError,"TaskSpec hash mismatch"):audit_double_replay(x)

if __name__=="__main__":unittest.main()
