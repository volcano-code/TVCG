"""Work-formula replay and backend dispatch, without claiming live mesh execution."""
from copy import deepcopy
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from . import acceptance as a


REPLAY = json.loads((Path(__file__).with_name("fixtures") / "work_acceptance_replay.json").read_text())


def request():
    return SimpleNamespace(center_of_mass_weight=REPLAY["beta"], design_vertices=np.eye(3),
        design_faces=np.array([[0, 1, 2]]), target_axes=np.eye(3), target_origin=np.zeros(3),
        fixed_shell_moments=np.zeros(10), material_density=1., mode="SPINNING_TOP",
        spinning_top_objective="INERTIA_AND_LOWERING", settings=SimpleNamespace(
            equation_profile="DUAL_MONOMIAL_RAW_V1", equation_tolerance=1e-12,
            physical_relative_tolerance=1e-6))


class JointReplay:
    """Archived vectors behind a fake backend; not a geometry acceptance run."""
    integral_profile = a.INTEGRAL_PROFILE
    rational_arithmetic_backend = "NATIVE_GMP_SURFACE_V1"

    def __init__(self, case="corrected"):
        self.request = request()
        self.case = REPLAY["cases"][case]
        self.joint_calls = self.value_calls = 0
        self.last_integral_evidence = None

    def packet(self, derivative):
        self.last_integral_evidence = dict(enclosure_certified=True, precision_bits=256,
                                          fallback_components=0, fixture_only=True)
        return dict(raw_moments=np.array(self.case["moments"]),
                    moment_jacobian=np.full((10, 10), np.nan) if derivative else None,
                    derivative_defined=derivative, geometry_hash="numeric-fixture-geometry",
                    topology_hash="numeric-fixture-topology",
                    boolean_backend_identity="cherchi-reference-numeric-fixture-not-native-proof")

    def __call__(self, q):
        self.joint_calls += 1
        return self.packet(True)


class ValueReplay(JointReplay):
    @property
    def moment_only_capabilities(self):
        return dict(schema="cherchi-native-moment-only-v1", independent_mesh_evaluation=True,
                    material_jacobian_avoided=True, intersection_motion_avoided=True,
                    source_motion_avoided=False, material_result_cache_used=False,
                    integral_profile=self.integral_profile)

    def evaluate_moments(self, q):
        self.value_calls += 1
        return self.packet(False)


@pytest.mark.parametrize("name", ["direct", "corrected"])
def test_archived_work_moments_keep_original_accept_and_reject(name):
    case = REPLAY["cases"][name]
    got = a.evaluate_moment_formula(case["moments"], case["q"], REPLAY["beta"])
    expected = case["expected"]
    assert got["accepted"] is expected["accepted"]
    assert got["scope"] == "moment_formula_diagnostic"
    assert got["independent_mesh_evaluated"] is False
    assert got["reference_verify_sha256"] == a.WORK_VERIFY_SHA256
    assert got["objective"] == pytest.approx(expected["objective"], rel=2e-15)
    assert got["mu"] == pytest.approx(expected["mu"], rel=2e-14)
    # Near 1e-13, LAPACK roundoff is material; the unchanged 1e-12 decision is tested above.
    assert got["raw_residual"] == pytest.approx(expected["raw_residual"], rel=2e-10, abs=3e-13)
    np.testing.assert_allclose(got["constraints"], expected["constraints"], rtol=0, atol=1e-20)
    np.testing.assert_allclose(got["center_of_mass"], expected["center_of_mass"], rtol=2e-15, atol=0)


def test_cached_symbolics_not_cached_moments_or_objective_weight(monkeypatch):
    import sympy
    a._compiled_formula.cache_clear()
    original = sympy.lambdify
    compilations = []

    def counted(*args, **kwargs):
        compilations.append(1)
        return original(*args, **kwargs)

    monkeypatch.setattr(sympy, "lambdify", counted)
    case = REPLAY["cases"]["corrected"]
    first = a.evaluate_moment_formula(case["moments"], case["q"], REPLAY["beta"])
    changed = np.array(case["moments"])
    changed[3] += .001
    second = a.evaluate_moment_formula(changed, case["q"], REPLAY["beta"])
    assert len(compilations) == 1
    assert second["objective"] != first["objective"]
    other_weight = a.evaluate_moment_formula(case["moments"], case["q"], REPLAY["beta"] + .1)
    assert len(compilations) == 2
    assert other_weight["objective"] - first["objective"] == pytest.approx(.1*case["moments"][3]/case["moments"][0])


def test_original_status_and_positive_multiplier_conditions_are_retained():
    case = REPLAY["cases"]["corrected"]
    assert not a.evaluate_moment_formula(case["moments"], case["q"], REPLAY["beta"], "RESOURCE_LIMIT_REACHED")["accepted"]
    flipped = a.evaluate_moment_formula(case["moments"], -np.array(case["q"]), REPLAY["beta"])
    assert flipped["mu"] < 0 and not flipped["accepted"]
    with pytest.raises(ValueError, match="nonzero"):
        a.evaluate_moment_formula(case["moments"], np.zeros(10), REPLAY["beta"])


def test_native_value_dispatch_keeps_fresh_evaluation_and_joint_fallback_is_explicit():
    value = ValueReplay()
    q = value.case["q"]
    first = a.verify(value, q, value.request, "CONVERGED")
    again = a.verify(value, q, value.request, "CONVERGED")
    assert value.value_calls == 2 and value.joint_calls == 0
    assert first["accepted"] and again["accepted"]
    assert first["verification_backend_mode"] == "native_moment_only"
    assert first["cut_jacobian_requested"] is False
    assert first["moment_only_capabilities"]["source_motion_avoided"] is False
    joint = JointReplay()
    fallback = a.verify(joint, q, joint.request, "CONVERGED")
    assert fallback["accepted"]  # The unused, fake NaN J cannot change this Moment-only formula.
    assert joint.joint_calls == 1
    assert fallback["verification_backend_mode"] == "joint_fallback"
    assert fallback["cut_jacobian_requested"] is True
    with pytest.raises(RuntimeError, match="capability unavailable"):
        a.verify(joint, q, joint.request, "CONVERGED", require_moment_only=True)
    assert joint.joint_calls == 1


def test_value_failure_propagates_without_changing_map_or_retrying_joint():
    value = ValueReplay()

    def failed(q):
        raise ValueError("native Boolean rejected")

    value.evaluate_moments = failed
    with pytest.raises(ValueError, match="Boolean rejected"):
        a.verify(value, value.case["q"], value.request, "CONVERGED")
    assert value.joint_calls == 0


def test_wrong_task_uncertified_integrals_and_surrogate_packets_cannot_pass():
    value = ValueReplay()
    q = value.case["q"]
    other = deepcopy(value.request)
    other.center_of_mass_weight += .01
    with pytest.raises(ValueError, match="request mismatch"):
        a.verify(value, q, other, "CONVERGED")
    assert value.value_calls == 0
    original = value.packet

    def uncertified(derivative):
        p = original(derivative)
        value.last_integral_evidence["enclosure_certified"] = False
        return p

    value.packet = uncertified
    with pytest.raises(ValueError, match="certified integral"):
        a.verify(value, q, value.request, "CONVERGED")

    def surrogate(derivative):
        p = original(derivative)
        p["boolean_backend_identity"] = "surrogate-no-Boolean"
        return p

    value.packet = surrogate
    with pytest.raises(ValueError, match="native Cherchi"):
        a.verify(value, q, value.request, "CONVERGED")


def test_certificate_from_an_earlier_query_is_not_reused():
    value = ValueReplay()
    saved = value.packet(False)
    value.evaluate_moments = lambda q: saved
    with pytest.raises(ValueError, match="certified integral"):
        a.verify(value, value.case["q"], value.request, "CONVERGED")


@pytest.mark.parametrize("field,value", [
    ("mode", "STATIC_BALANCE"), ("spinning_top_objective", "LOWER_CENTER_OF_MASS"),
    ("equation_profile", "LEGACY_NATIVE_NORMALIZED_V1"), ("equation_tolerance", 1e-6),
    ("physical_relative_tolerance", 1e-3),
])
def test_other_objectives_equation_profiles_or_thresholds_are_not_silently_accepted(field, value):
    backend = ValueReplay()
    target = backend.request if field in ("mode", "spinning_top_objective") else backend.request.settings
    setattr(target, field, value)
    with pytest.raises(ValueError, match="spinning-top acceptance profile"):
        a.verify(backend, backend.case["q"], backend.request, "CONVERGED")
    assert backend.joint_calls == backend.value_calls == 0
