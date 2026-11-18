import unittest
from research.NeuralMoments.r3_native_independent_moment_compare_v1 import compare_records, MOMENT_ORDER

def pair():
    raw=[2.0,0.5,0.5,0.5,0.2,0.2,0.2,0.01,0.01,0.01]
    meta={"case_id":"case-1","geometry_hash":"geom-1","coordinate_frame":"world",
      "length_unit":"m","meters_per_unit":1.0,"density":1000.0,"moment_order":MOMENT_ORDER}
    native={"ledger":{"moment_metadata":meta},"native_result":{"final_total_raw_moments":raw}}
    independent={"case_id":"case-1","source_geometry_hash":"geom-1","coordinate_frame":"world",
      "length_unit":"m","meters_per_unit":1.0,"density":1000.0,
      "moment_order":MOMENT_ORDER,"backend_identity":"independent_tetra_exact_v1","raw_moments":raw}
    return native,independent

class R3NativeIndependentMomentCompareTests(unittest.TestCase):
    def test_exact_match(self):
        n,i=pair(); r=compare_records(n,i)
        self.assertTrue(r["moment_match"]);self.assertTrue(all(r["component_pass"]))
    def test_componentwise_tolerance_catches_small_moment_difference(self):
        n,i=pair();i["raw_moments"][9]+=1e-5
        r=compare_records(n,i,rtol=1e-8,atol=1e-14)
        self.assertFalse(r["moment_match"]);self.assertFalse(r["component_pass"][9])
    def test_geometry_mismatch_rejected(self):
        n,i=pair();i["source_geometry_hash"]="other"
        with self.assertRaisesRegex(ValueError,"geometry hash"):compare_records(n,i)
    def test_frame_mismatch_rejected(self):
        n,i=pair();i["coordinate_frame"]="local"
        with self.assertRaisesRegex(ValueError,"coordinate_frame"):compare_records(n,i)
    def test_density_mismatch_rejected(self):
        n,i=pair();i["density"]=998
        with self.assertRaisesRegex(ValueError,"density mismatch"):compare_records(n,i)
    def test_unit_scale_mismatch_rejected(self):
        n,i=pair();i["meters_per_unit"]=0.001
        with self.assertRaisesRegex(ValueError,"meters_per_unit mismatch"):compare_records(n,i)
    def test_backend_must_be_independent(self):
        n,i=pair();i["backend_identity"]="native_solver"
        with self.assertRaisesRegex(ValueError,"distinct"):compare_records(n,i)
    def test_missing_native_metadata_rejected(self):
        n,i=pair();n["ledger"].pop("moment_metadata")
        with self.assertRaisesRegex(ValueError,"moment_metadata"):compare_records(n,i)

if __name__=="__main__":unittest.main()
