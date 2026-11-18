import unittest, math
from research.NeuralMoments.r3_native_solver_replay_v1 import validate_native_replay_inputs
from research.NeuralMoments.r3_faster_initialization_v1 import FasterInitialization, SCHEMA

def init_record():
    v=[float(i+1) for i in range(10)]; n=math.sqrt(sum(x*x for x in v)); v=[x/n for x in v]
    return FasterInitialization.from_dict({
      "schema":SCHEMA,"mode":"spinning_top","equation_profile":"dual_monomial_raw_v1",
      "deterministic_seed":7,"maximum_iterations":60,"maximum_restarts":8,
      "maximum_function_evaluations_per_restart":60,"equation_tolerance":1e-12,
      "physical_relative_tolerance":1e-6,"line_search_reduction":0.5,
      "maximum_geodesic_step":0.5,"initial_guesses":[v],
      "initialization_source":"test-explicit-vector","solver_source_commit":"abc",
      "evaluator_identity":"evaluator-v1","geometry_hash":"geometry-1","task_hash":"task-1"
    })

def request_payload():
    return {
      "case_id":"case-1","geometry_hash":"geometry-1","task_hash":"task-1",
      "outer_vertices":[[0,0,0],[1,0,0],[0,1,0],[0,0,1]],
      "outer_faces":[[0,1,2],[0,1,3],[0,2,3],[1,2,3]],
      "design_vertices":[[0,0,0],[1,0,0],[0,1,0],[0,0,1]],
      "design_faces":[[0,1,2],[0,1,3],[0,2,3],[1,2,3]],
      "fixed_shell_moments":[0.0]*10,"target_origin":[0,0,0],
      "target_axes":[[1,0,0],[0,1,0],[0,0,1]],"mode":"SPINNING_TOP",
      "material_density":1.0,"original_units_per_normalized_unit":1.0,
      "minimum_wall_thickness":0.0,"spinning_top_objective":"INERTIA_AND_LOWERING",
      "settings":{"maximum_iterations":60,"maximum_restarts":8,
        "maximum_function_evaluations_per_restart":60,"equation_tolerance":1e-12,
        "physical_relative_tolerance":1e-6,"deterministic_seed":7,
        "equation_profile":"DUAL_MONOMIAL_RAW_V1","capture_evaluation_payload":True}
    }

class R3NativeReplayIntegrationTests(unittest.TestCase):
    def test_request_init_evaluator_contract_integrates(self):
        req,evaluator=validate_native_replay_inputs(request_payload(),init_record(),
          {"evaluator_identity":"evaluator-v1","settings":{}})
        self.assertEqual(req.mode,"SPINNING_TOP")
        self.assertTrue(req.settings.capture_evaluation_payload)
        self.assertEqual(evaluator.derivative_method,"LEGACY_CONTINUOUS_IMPLICIT_SHAPE")

    def test_geometry_hash_mismatch_rejected(self):
        p=request_payload();p["geometry_hash"]="wrong"
        with self.assertRaisesRegex(ValueError,"geometry_hash"):validate_native_replay_inputs(p,init_record(),{"settings":{}})

    def test_task_hash_mismatch_rejected(self):
        p=request_payload();p["task_hash"]="wrong"
        with self.assertRaisesRegex(ValueError,"task_hash"):validate_native_replay_inputs(p,init_record(),{"settings":{}})

    def test_evaluator_identity_mismatch_rejected(self):
        with self.assertRaisesRegex(ValueError,"evaluator identity"):
            validate_native_replay_inputs(request_payload(),init_record(),
              {"evaluator_identity":"different","settings":{}})

    def test_iteration_budget_mismatch_rejected(self):
        p=request_payload();p["settings"]["maximum_iterations"]=20
        with self.assertRaisesRegex(ValueError,"maximum_iterations"):
            validate_native_replay_inputs(p,init_record(),{"settings":{}})

    def test_payload_capture_mandatory(self):
        p=request_payload();p["settings"]["capture_evaluation_payload"]=False
        with self.assertRaisesRegex(ValueError,"capture_evaluation_payload"):
            validate_native_replay_inputs(p,init_record(),{"settings":{}})

if __name__=="__main__":unittest.main()
