import unittest
import numpy as np
from research.NeuralMoments.r3_independent_tetra_moments_v1 import integrate_tetrahedra, derive_mass_properties, MOMENT_ORDER
from research.NeuralMoments.r3_independent_physical_verification_v1 import verify_spinning_top_constraints

def record_for_tetra():
    tet=np.array([[[0,0,0],[1,0,0],[0,1,0],[0,0,1]]],float)
    raw=integrate_tetrahedra(tet,density=2.0,meters_per_unit=1.0)
    props=derive_mass_properties(raw)
    # Pick a right-handed principal frame; z is the first principal axis.
    eigvals,eigvecs=np.linalg.eigh(np.asarray(props["inertia_center"]))
    R=eigvecs.copy()
    if np.linalg.det(R)<0: R[:,0]*=-1
    com=np.asarray(props["center_of_mass"])
    return {"case_id":"analytic-tetra","source_geometry_hash":"geom-1",
      "coordinate_frame":"world","length_unit":"m","meters_per_unit":1.0,
      "density":2.0,"geometric_volume_m3":1/6,"density_source":"test analytical density",
      "tetrahedra_sha256":"tetra-sha","backend_sha256":"backend-sha","backend_identity":"independent_tetra_exact_v1",
      "moment_order":MOMENT_ORDER,"raw_moments":raw.tolist()}, {
        "origin":com.tolist(),"x_axis":R[:,0].tolist(),"y_axis":R[:,1].tolist(),
        "axis":R[:,2].tolist(),"coordinate_frame":"world","length_unit":"m",
        "meters_per_unit":1.0,"density":2.0,"geometry_hash":"geom-1"}

TOL={"Lx_abs":1e-12,"Ly_abs":1e-12,"Icxz_abs":1e-12,"Icyz_abs":1e-12,"mass_relative":1e-12}

class R3IndependentPhysicalVerificationTests(unittest.TestCase):
    def test_principal_frame_analytic_tetra_passes(self):
        rec,frame=record_for_tetra()
        r=verify_spinning_top_constraints(rec,frame,TOL,expected_geometry_hash="geom-1")
        self.assertTrue(r["hard_constraints_pass"])
        self.assertTrue(r["overall_pass"])
        self.assertLess(r["diagnostics"]["com_axis_offset_m"],1e-12)

    def test_wrong_axis_frame_fails_constraints(self):
        rec,frame=record_for_tetra()
        frame["axis"]=[0,0,1];frame["x_axis"]=[1,0,0];frame["y_axis"]=[0,1,0]
        r=verify_spinning_top_constraints(rec,frame,TOL)
        self.assertFalse(r["hard_constraints_pass"])

    def test_mass_density_volume_mismatch_fails(self):
        rec,frame=record_for_tetra();rec["geometric_volume_m3"]=0.2
        r=verify_spinning_top_constraints(rec,frame,TOL)
        self.assertFalse(r["hard_constraint_passes"]["mass_relative"])

    def test_wrong_geometry_hash_rejected(self):
        rec,frame=record_for_tetra()
        with self.assertRaisesRegex(ValueError,"geometry hash mismatch"):
            verify_spinning_top_constraints(rec,frame,TOL,expected_geometry_hash="other")

    def test_nonorthonormal_task_frame_rejected(self):
        rec,frame=record_for_tetra();frame["y_axis"]=frame["x_axis"]
        with self.assertRaisesRegex(ValueError,"right-handed orthonormal"):
            verify_spinning_top_constraints(rec,frame,TOL)

    def test_missing_explicit_tolerance_rejected(self):
        rec,frame=record_for_tetra();tol=dict(TOL);tol.pop("Icxz_abs")
        with self.assertRaisesRegex(ValueError,"tolerances missing"):
            verify_spinning_top_constraints(rec,frame,tol)

    def test_wrong_moment_order_rejected(self):
        rec,frame=record_for_tetra();rec["moment_order"]=["V"]*10
        with self.assertRaisesRegex(ValueError,"moment_order mismatch"):
            verify_spinning_top_constraints(rec,frame,TOL)

if __name__=="__main__":unittest.main()
