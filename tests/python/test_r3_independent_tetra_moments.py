import unittest
import numpy as np
from research.NeuralMoments.r3_independent_tetra_moments_v1 import (
    integrate_tetrahedra, derive_mass_properties
)

class R3IndependentTetraMomentTests(unittest.TestCase):
    def setUp(self):
        self.tet=np.array([[[0,0,0],[1,0,0],[0,1,0],[0,0,1]]],dtype=float)

    def test_unit_tetra_exact_ten_moments(self):
        m=integrate_tetrahedra(self.tet,density=1.0,meters_per_unit=1.0)
        expected=np.array([1/6,1/24,1/24,1/24,1/60,1/60,1/60,1/120,1/120,1/120])
        np.testing.assert_allclose(m,expected,rtol=1e-13,atol=1e-15)

    def test_density_and_unit_scale(self):
        m=integrate_tetrahedra(self.tet,density=2.0,meters_per_unit=1e-3)
        self.assertAlmostEqual(m[0],2*(1/6)*1e-9,places=20)
        self.assertAlmostEqual(m[1],2*(1/24)*1e-12,places=22)

    def test_negative_orientation_fails_closed(self):
        t=self.tet.copy();t[:,[1,2]]=t[:,[2,1]]
        with self.assertRaisesRegex(ValueError,"negative signed volume"):
            integrate_tetrahedra(t,density=1,meters_per_unit=1)

    def test_degenerate_tetra_fails_closed(self):
        t=self.tet.copy();t[0,3]=[1,1,0]
        with self.assertRaisesRegex(ValueError,"degenerate"):
            integrate_tetrahedra(t,density=1,meters_per_unit=1)

    def test_nonfinite_fails_closed(self):
        t=self.tet.copy();t[0,0,0]=np.nan
        with self.assertRaisesRegex(ValueError,"non-finite"):
            integrate_tetrahedra(t,density=1,meters_per_unit=1)

    def test_mass_properties(self):
        m=integrate_tetrahedra(self.tet,density=1,meters_per_unit=1)
        p=derive_mass_properties(m)
        np.testing.assert_allclose(p["center_of_mass"],[0.25]*3,atol=1e-14)
        I=np.asarray(p["inertia_center"])
        np.testing.assert_allclose(I,I.T,atol=1e-14)
        self.assertTrue(np.linalg.eigvalsh(I).min()>0)

if __name__=="__main__":unittest.main()
