import unittest
from research.NeuralMoments.r3_solver_event_ledger_v1 import SCHEMA, validate_ledger, scientific_hash

def payload():
    return {"schema":SCHEMA,"case_id":"case-01","geometry_hash":"geom-hash","task_hash":"task-hash",
      "evaluator_identity":"exact-map-v1","solver_source_commit":"abc123",
      "initialization_record_hash":"init-hash","equation_profile":"legacy_native_normalized_v1",
      "reported_function_evaluations":3,"reported_restart_function_evaluations":[2,1],
      "terminal_status":"converged","events":[
        {"type":"attempt_start","restart":0,"evaluation_index":0,"outcome":"started"},
        {"type":"evaluation","restart":0,"evaluation_index":1},
        {"type":"evaluation","restart":0,"evaluation_index":2},
        {"type":"attempt_terminal","restart":0,"evaluation_index":2,"outcome":"failed"},
        {"type":"attempt_start","restart":1,"evaluation_index":2,"outcome":"started"},
        {"type":"evaluation","restart":1,"evaluation_index":3},
        {"type":"attempt_terminal","restart":1,"evaluation_index":3,"outcome":"converged"}]}

class R3SolverEventLedgerTests(unittest.TestCase):
    def test_valid_restart_and_evaluation_accounting(self):
        r=validate_ledger(payload())
        self.assertTrue(r["valid"]); self.assertEqual(r["restart_evaluations"],[2,1])
    def test_missing_terminal_rejected(self):
        p=payload(); p["events"].pop()
        with self.assertRaisesRegex(ValueError,"one start and terminal"): validate_ledger(p)
    def test_evaluation_count_mismatch_rejected(self):
        p=payload(); p["reported_function_evaluations"]=4
        with self.assertRaisesRegex(ValueError,"event count"): validate_ledger(p)
    def test_restart_gap_rejected(self):
        p=payload(); p["events"][4]["restart"]=2; p["events"][5]["restart"]=2; p["events"][6]["restart"]=2
        with self.assertRaisesRegex(ValueError,"contiguous"): validate_ledger(p)
    def test_scientific_hash_excludes_timing(self):
        p=payload(); a=scientific_hash(p); p["wall_seconds"]=123.4
        p["events"][1]["residual_evaluation_seconds"]=0.2
        self.assertEqual(a,scientific_hash(p))
    def test_scientific_hash_binds_decisions(self):
        p=payload(); a=scientific_hash(p); p["events"][3]["reason"]="different"
        self.assertNotEqual(a,scientific_hash(p))
    def test_convergence_must_be_final_restart(self):
        p=payload(); p["events"][3]["outcome"]="converged"
        with self.assertRaisesRegex(ValueError,"continues after convergence"): validate_ledger(p)

if __name__=="__main__": unittest.main()
