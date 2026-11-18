import json, math, tempfile, unittest
from pathlib import Path
from research.NeuralMoments.r3_faster_initialization_v1 import SCHEMA, FasterInitialization

def unit(v):
    n=math.sqrt(sum(x*x for x in v)); return [x/n for x in v]

class R3FasterInitializationTests(unittest.TestCase):
    def record(self):
        return {"schema":SCHEMA,"mode":"spinning_top","equation_profile":"legacy_native_normalized_v1",
          "deterministic_seed":42,"maximum_iterations":60,"maximum_restarts":8,
          "maximum_function_evaluations_per_restart":60,"equation_tolerance":1e-12,
          "physical_relative_tolerance":1e-6,"line_search_reduction":0.5,
          "maximum_geodesic_step":0.5,"initial_guesses":[unit([1,2,3,4,5,6,7,8,9,10])],
          "initialization_source":"deterministicInitialQuadrics","solver_source_commit":"abc123",
          "evaluator_identity":"exact-moment-map-v1","geometry_hash":"geometry-sha256",
          "task_hash":"task-sha256"}

    def test_roundtrip(self):
        x=FasterInitialization.from_dict(self.record())
        self.assertEqual(len(x.initial_guesses),1)
        self.assertAlmostEqual(sum(v*v for v in x.initial_guesses[0]),1.0)

    def test_physical_geometry_not_confused_with_beta(self):
        x=FasterInitialization.from_dict(self.record())
        self.assertNotIn("original_mesh",x.to_dict())
        self.assertIn("initial_guesses",x.to_dict())

    def test_non_unit_guess_rejected(self):
        r=self.record(); r["initial_guesses"]=[[1.0]+[0.0]*9]
        r["initial_guesses"][0][0]=2.0
        with self.assertRaisesRegex(ValueError,"unit norm"): FasterInitialization.from_dict(r)

    def test_paper_line_search_constants_are_locked(self):
        r=self.record(); r["line_search_reduction"]=0.25
        with self.assertRaisesRegex(ValueError,"locks"): FasterInitialization.from_dict(r)

    def test_missing_initialization_provenance_rejected(self):
        r=self.record(); r.pop("initialization_source")
        with self.assertRaisesRegex(ValueError,"incomplete"): FasterInitialization.from_dict(r)

    def test_strict_dual_eval_budget(self):
        r=self.record(); r["equation_profile"]="dual_monomial_raw_v1"; r["maximum_function_evaluations_per_restart"]=61
        with self.assertRaisesRegex(ValueError,"at most 60"): FasterInitialization.from_dict(r)

if __name__=="__main__": unittest.main()
