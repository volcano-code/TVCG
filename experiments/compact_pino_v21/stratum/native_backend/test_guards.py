"""Source-only negative controls for the versioned native rank-chart entry."""
import json
from pathlib import Path
from types import SimpleNamespace
import unittest

import numpy as np

from bridge import load_native


class SourceGuards(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.ext = load_native()
        profile = Path('/data/xvxinyao/TVCG-phase2-12-operator/phase3_native_teacher/production_profile.json')
        cls.settings = json.loads(profile.read_text())
        cls.design = np.array(np.meshgrid(*[[-.5,.5]]*3)).reshape(3,-1).T.copy()
        cls.plane = np.array([.02,.3,.4,.8,0.,0.,0.,0.,0.,0.])
        cls.repeated = np.array([0.,0.,0.,.3,.5,.5,0.,0.,0.,0.])

    def value(self, q, **changes):
        return self.ext.source_value(q, self.design, SimpleNamespace(**(self.settings|changes)))

    def test_flags_and_rank_rejection(self):
        for changes in (dict(algebraic_finite_map=False),dict(fixed_normal_paraboloid=True),
                        dict(rational_source_coordinates=True),dict(compact_rational_source_coordinates=True)):
            with self.assertRaisesRegex(ValueError,'binary64 algebraic finite-map'):
                self.value(self.plane,**changes)
        near_plane = self.plane.copy()
        near_plane[4] = 1.e-18
        with self.assertRaisesRegex(ValueError,'exactly zero'):
            self.value(near_plane)
        near_rank_two = self.repeated.copy()
        near_rank_two[6] = 1.e-13
        with self.assertRaisesRegex(ValueError,'roundoff-scale'):
            self.value(near_rank_two)
        ellipsoid = np.array([-.1,0,0,0,1,1,1,0,0,0], dtype=float)
        with self.assertRaisesRegex(ValueError,'only R3 planes and R8 paraboloids'):
            self.value(ellipsoid)

    def test_value_at_derivative_seam(self):
        # Repeated nonzero curvatures admit a deterministic value. This
        # eigenframe construction has no unique smooth chart motion there.
        value = self.value(self.repeated)
        motion = self.ext.source_motion(self.repeated,self.design,SimpleNamespace(**self.settings))
        self.assertTrue(value['defined'] and value['side_selection_defined'])
        self.assertFalse(motion['defined'])
        self.assertIn('separated eigenspectrum',motion['diagnostic'])
        self.assertNotIn('vertex_velocity',motion)
        self.assertNotIn('vertex_velocity',value)
        np.testing.assert_array_equal(value['vertices'],motion['vertices'])

    def test_remote_caps_and_domain_coverage(self):
        with self.assertRaisesRegex((ValueError,RuntimeError),'enclose design box'):
            self.value(self.plane,truncation_radius=.1)
        far = self.plane.copy()
        far[0] = 500
        with self.assertRaisesRegex((ValueError,RuntimeError),'growth budget'):
            self.value(far,maximum_radius_doublings=0)


if __name__=='__main__':
    unittest.main()
