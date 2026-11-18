import unittest
from research.NeuralMoments.r3_deterministic_replay_audit_v1 import (
    MOMENT_ORDER, compare_replays, derive_spinning_top_physics, verify_independent_moments
)

def bundle(event="ev",trace="tr",quad=None):
    ledger={"case_id":"case","geometry_hash":"geom","task_hash":"task",
      "evaluator_identity":"oracle","solver_source_commit":"abc",
      "initialization_record_hash":"init","equation_profile":"dual_monomial_raw_v1"}
    native={"event_hash":event,"trace_hash":trace,"input_hash":"in","output_hash":"out",
      "status":"CONVERGED","function_evaluations":4,"restart_function_evaluations":[4],
      "quadric_coefficients":quad or [1.0]+[0.0]*9}
    return {"ledger":ledger,"native_result":native}

class R3DeterministicReplayAuditTests(unittest.TestCase):
    def test_identical_replays_pass(self):
        r=compare_replays(bundle(),bundle())
        self.assertTrue(r["deterministic"]); self.assertEqual(r["mismatches"],[])
    def test_event_hash_difference_fails(self):
        r=compare_replays(bundle(),bundle(event="different"))
        self.assertFalse(r["deterministic"]); self.assertIn("native_event_hash",r["mismatches"])
    def test_initial_or_final_quadric_difference_fails(self):
        r=compare_replays(bundle(),bundle(quad=[0.0,1.0]+[0.0]*8))
        self.assertIn("quadric_coefficients",r["mismatches"])
    def test_timing_is_not_part_of_signature(self):
        a,b=bundle(),bundle()
        a["native_result"]["wall_seconds"]=1.0; b["native_result"]["wall_seconds"]=2.0
        self.assertTrue(compare_replays(a,b)["deterministic"])
    def test_independent_moment_match_and_physics(self):
        raw=[1.0,0,0,0,0.2,0.2,0.3,0,0,0]
        record={"moment_order":list(MOMENT_ORDER),"raw_moments":raw,
          "backend_identity":"separate-oracle-v1","source_geometry_hash":"geom"}
        r=verify_independent_moments(raw,record,density=1.0)
        self.assertTrue(r["moment_pass"]); self.assertTrue(r["physical_constraint_pass"])
    def test_moment_mismatch_detected(self):
        native=[1.0,0,0,0,0.2,0.2,0.3,0,0,0]
        other=[1.0,0,0,0,0.21,0.2,0.3,0,0,0]
        record={"moment_order":list(MOMENT_ORDER),"raw_moments":other,
          "backend_identity":"separate-oracle-v1","source_geometry_hash":"geom"}
        r=verify_independent_moments(native,record,density=1.0)
        self.assertFalse(r["moment_pass"])
    def test_unknown_moment_order_rejected(self):
        record={"moment_order":["wrong"]*10,"raw_moments":[1.0]+[0.0]*9,
          "backend_identity":"oracle","source_geometry_hash":"geom"}
        with self.assertRaisesRegex(ValueError,"moment_order"):
            verify_independent_moments([1.0]+[0.0]*9,record,density=1.0)

if __name__=="__main__": unittest.main()
