"""Actual Python exact integration on a manufactured Boolean packet.

The native Boolean/source is a deterministic fixture. These tests demonstrate
control flow and exact value equivalence, not deployed native-mesh speed or
GMP/MPFR runtime certification.
"""
from fractions import Fraction
from pathlib import Path
from types import SimpleNamespace
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "python"))
from make_it_stand.verification import cherchi_inprocess as ci


VERTICES = np.array([[0., 0., 0.], [1., 0., 0.], [0., 1., 0.], [0., 0., 1.]])
FACES = np.array([[0, 2, 1], [0, 1, 3], [0, 3, 2], [1, 2, 3]])
Q = np.array([1., .2, .3, .4, .1, .1, .1, 0., 0., 0.])


def fixture_evaluator():
    obj = ci.InProcessCherchiEvaluator.__new__(ci.InProcessCherchiEvaluator)
    obj.request = SimpleNamespace(target_axes=np.eye(3), target_origin=np.zeros(3),
                                  design_faces=FACES, material_density=1.)
    obj.design = VERTICES.copy()
    obj.target_design = VERTICES.copy()
    obj.length = 1.
    obj.normalization_origin = np.zeros(3)
    obj.fixed = np.array([.1, 0., 0., .02, 0., 0., .004, 0., 0., 0.])
    source = dict(defined=True, side_selection_defined=True, diagnostic="", interior_is_positive=True,
                  vertices=np.array([[-1., -1., -1.], [5., -1., -1.], [-1., 5., -1.], [-1., -1., 5.]]),
                  faces=FACES, surface_type="ELLIPSOID", cap_face_count=0,
                  source_normal_edges=np.full((4, 2), -1), vertex_velocity=np.zeros((4, 3, 10)))
    obj.source_fixture = source
    obj.native = SimpleNamespace(
        evaluate_spin_it_faster_source_motion=lambda *args: source,
        audit_exact_triangle_boundary=lambda *args: dict(boundary_valid=True))
    obj.triangulation = object()
    tokens = [[f"{int(x)}/1" for x in row] for row in VERTICES]
    packet = dict(backend="cherchi-reference-7bd6c2697695f555f9a15d7e154e084d841316f0-inprocess-v1",
                  vertices=VERTICES, faces=FACES, exact_vertices=tokens,
                  construction_records=[dict(vertex_index=i, point_type="EXPLICIT3D", parameter_t=None,
                    operand_coordinates_exact=[row]) for i, row in enumerate(tokens)])
    obj.bridge = SimpleNamespace(boolean=lambda *args: packet)
    obj.stage_observer = None
    obj.call_counts = dict(source=0, boolean=0, integration=0, derivative=0)
    obj.stage_seconds = {}
    obj._materials = {}
    obj.arithmetic = None
    obj.arithmetic_identity = None
    obj.integral_profile = "EXACT_GMP_V1"
    obj.rational_arithmetic_backend = "PYTHON_FRACTION_V1"
    obj.branch_identity_policy = "SOURCE_FEATURE_BRANCH_V2"
    obj.context_identity = "manufactured-original-tetrahedron"
    obj.backend_identity = "cherchi-reference-fixture-not-native-proof"
    obj.last = None
    return obj


def test_value_entry_repeats_boolean_and_integrates_exactly_without_motion(monkeypatch):
    be = fixture_evaluator()
    joint = be(Q)
    assert joint["derivative_defined"]
    np.testing.assert_array_equal(joint["moment_jacobian"], np.zeros((10, 10)))
    assert be.call_counts["intersection_motion"] == be.call_counts["derivative"] == 1

    def forbidden(*args, **kwargs):
        raise AssertionError("Moment-only evaluated a cut derivative")

    monkeypatch.setattr(ci, "recover_construction_vertex_motion", forbidden)
    be._differentiate = forbidden
    first = be.evaluate_moments(Q)
    second = be.evaluate_moments(Q)
    # Repeated final acceptance calls are fresh Booleans, not a q-result cache.
    assert be.call_counts["boolean"] == be.call_counts["source"] == be.call_counts["integration"] == 3
    assert be.call_counts["physical_source_boundary"] == 3
    assert be.call_counts["intersection_motion"] == be.call_counts["derivative"] == 1
    expected = np.array([1/6, 1/24, 1/24, 1/24, 1/60, 1/60, 1/60, 1/120, 1/120, 1/120])
    np.testing.assert_allclose(first["variable_raw_moments"], expected, rtol=0, atol=1e-17)
    for key in ("raw_moments", "variable_raw_moments"):
        np.testing.assert_array_equal(first[key], joint[key])
        np.testing.assert_array_equal(first[key], second[key])
    for key in ("geometry_hash", "topology_hash", "evaluation_map_identity", "boolean_backend_identity"):
        assert first[key] == joint[key]
    assert first["moment_jacobian"] is None and first["derivative_defined"] is False
    assert first["moment_only_capabilities"]["material_jacobian_avoided"]
    assert first["moment_only_capabilities"]["source_motion_avoided"] is False
    saved = be.material(Q)
    assert saved["moment_jacobian"] is None and saved["derivative_defined"] is False


def test_value_entry_keeps_same_source_and_boolean_failures():
    be = fixture_evaluator()
    be.source_fixture["defined"] = False
    with pytest.raises(ValueError, match="SOURCE_OR_SELECTOR_UNDEFINED"):
        be.evaluate_moments(Q)
    assert be.call_counts["boolean"] == 0
    be.source_fixture["defined"] = True
    be.bridge.boolean = lambda *args: dict(backend="different-kernel")
    with pytest.raises(ValueError, match="CHERCHI_BACKEND_IDENTITY_MISMATCH"):
        be.evaluate_moments(Q)


def test_value_entry_keeps_native_certified_integral_route():
    be = fixture_evaluator()
    be.integral_profile = "CERTIFIED_MPFR256_GMP_FALLBACK_V1"
    be.rational_arithmetic_backend = "NATIVE_GMP_SURFACE_V1"
    operations = []

    class Prepared:
        def __init__(self, tokens, faces):
            operations.append("prepared")
            self.rational = tuple(tuple(Fraction(v) for v in row) for row in tokens)
            self.faces = faces

        def integrate_moments_certified(self, density):
            operations.append("certified")
            # Only a protocol fixture: arithmetic uses actual Python Fractions.
            m = ci.integrate_rational_polyhedral_raw_moments(VERTICES, self.faces,
                density=density, rational_vertices=self.rational)
            return m, dict(enclosure_certified=True, precision_bits=256, fixture_only=True)

    be.arithmetic = SimpleNamespace(PreparedBoundary=Prepared)
    out = be.evaluate_moments(Q)
    assert operations == ["prepared", "certified"]
    assert be.last_integral_evidence["fixture_only"] is True
    assert out["moment_jacobian"] is None
    assert be.call_counts["derivative"] == 0
