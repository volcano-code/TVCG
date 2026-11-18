import math, unittest
from research.NeuralMoments.r3_native_result_acceptance_v1 import validate_native_result

def result():
    return {"status":"CONVERGED","diagnostic":"","function_evaluations":3,"restart_count":1,
      "restart_function_evaluations":[3],"events":[{"type":"evaluation"}],"trace":[],
      "event_hash":"e","trace_hash":"t","input_hash":"i","output_hash":"o",
      "final_geometry_verified":True,"verification_diagnostic":"pass",
      "final_total_raw_moments":[1.0]+[0.0]*9,
      "quadric_coefficients":[1.0]+[0.0]*9,"final_geometry_hash":"geom-final"}

class R3NativeResultAcceptanceTests(unittest.TestCase):
    def test_converged_and_geometry_verified_accepted(self):
        r=validate_native_result(result())
        self.assertTrue(r["physically_accepted"])
    def test_nonconverged_not_accepted(self):
        x=result();x["status"]="MAX_ITERATIONS"
        self.assertFalse(validate_native_result(x)["physically_accepted"])
    def test_geometry_failure_not_accepted(self):
        x=result();x["final_geometry_verified"]=False
        self.assertFalse(validate_native_result(x)["physically_accepted"])
    def test_restart_evaluation_accounting(self):
        x=result();x["restart_function_evaluations"]=[2]
        with self.assertRaisesRegex(ValueError,"do not sum"):validate_native_result(x)
    def test_nonfinite_moment_rejected(self):
        x=result();x["final_total_raw_moments"][0]=float("nan")
        with self.assertRaisesRegex(ValueError,"non-finite"):validate_native_result(x)
    def test_nonunit_quadric_rejected(self):
        x=result();x["quadric_coefficients"][0]=2.0
        with self.assertRaisesRegex(ValueError,"unit norm"):validate_native_result(x)
    def test_missing_final_geometry_hash_rejected(self):
        x=result();x["final_geometry_hash"]=""
        with self.assertRaisesRegex(ValueError,"empty"):validate_native_result(x)

if __name__=="__main__":unittest.main()
