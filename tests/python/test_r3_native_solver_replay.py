import unittest
from research.NeuralMoments.r3_native_solver_replay_v1 import _jsonable, _request
from make_it_stand.spin import SpinSolverConfig

class R3NativeReplayBindingTests(unittest.TestCase):
    def test_request_metadata_is_removed_before_native_request_construction(self):
        payload={
          "case_id":"m69-case-1","geometry_hash":"geom","task_hash":"task",
          "outer_vertices":[[0,0,0],[1,0,0],[0,1,0],[0,0,1]],
          "outer_faces":[[0,1,2],[0,1,3],[0,2,3],[1,2,3]],
          "design_vertices":[[0,0,0],[1,0,0],[0,1,0],[0,0,1]],
          "design_faces":[[0,1,2],[0,1,3],[0,2,3],[1,2,3]],
          "mode":"SPINNING_TOP",
          "settings":{"equation_profile":"DUAL_MONOMIAL_RAW_V1","capture_evaluation_payload":True}
        }
        request=_request(payload)
        self.assertEqual(request.mode,"SPINNING_TOP")
        self.assertTrue(request.settings.capture_evaluation_payload)
        self.assertIsInstance(request.settings,SpinSolverConfig)

    def test_native_result_json_conversion(self):
        self.assertEqual(_jsonable({"x":(1,2),"y":[3]}),{"x":[1,2],"y":[3]})

if __name__=="__main__": unittest.main()
