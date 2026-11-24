"""Check the eigensolver reporting allowance on either supported standalone interpreter."""
from copy import deepcopy
import math
import unittest

from .archive import verify_summary_groups


class SummaryPortability(unittest.TestCase):
    def setUp(self):
        self.query = [0.] * 10
        self.query[5:7] = [0.6, 0.8]
        self.rows = [{"dimension": 8, "geometry_id": "unit", "query": self.query,
                      "steps": [{"columns": [{"endpoints": [{"query": self.query}]}]}]}]
        self.center = "center_nearest_quadratic_eigenvalue_absolute"
        self.drift = "endpoint_nearest_eigenvalue_drift_absolute"
        self.recorded = {"all": {"counts": {"passed": 1}, "maxima": {
            self.center: 2.22e-16, self.drift: 6.8e-16, "another_float": 0.125}}}

    def test_two_near_zero_eigenvalue_reports_can_round_differently(self):
        recomputed = deepcopy(self.recorded)
        recomputed["all"]["maxima"].update({self.center: 1.11e-16, self.drift: 5.32e-16})
        verify_summary_groups(self.recorded, recomputed, self.rows)

    def test_every_other_float_and_count_remains_exact(self):
        for key, value in (("another_float", math.nextafter(0.125, 1.0)), ("passed", True), ("passed", 0)):
            with self.subTest(key=key, value=value):
                rec = deepcopy(self.recorded)
                section = "counts" if key == "passed" else "maxima"
                rec["all"][section][key] = value
                with self.assertRaises(ValueError):
                    verify_summary_groups(rec, self.recorded, self.rows)

    def test_no_absolute_floor_for_zero_quadratic_blocks(self):
        self.query[5:7] = [0., 0.]
        recomputed = deepcopy(self.recorded)
        recomputed["all"]["maxima"][self.drift] = math.nextafter(6.8e-16, 1.)
        with self.assertRaisesRegex(ValueError, "rounding allowance"):
            verify_summary_groups(self.recorded, recomputed, self.rows)

    def test_too_large_or_negative_eigenvalue_report_is_rejected(self):
        for value in (1e-13, -1e-16):
            with self.subTest(value=value):
                changed = deepcopy(self.recorded)
                changed["all"]["maxima"][self.center] = value
                with self.assertRaisesRegex(ValueError, "rounding allowance"):
                    verify_summary_groups(changed, self.recorded, self.rows)
