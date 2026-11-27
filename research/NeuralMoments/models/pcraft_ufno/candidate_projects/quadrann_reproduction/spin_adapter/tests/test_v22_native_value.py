"""Source dispatch controls; fixtures grant no native numerical qualification."""
from types import SimpleNamespace

import numpy as np
import pytest

from research.NeuralMoments.models.pcraft_ufno.candidate_projects.quadrann_reproduction.spin_adapter.v21_native import (
    RawChart, native_operand, native_operand_capabilities, operand_from_value_source,
)


Q = np.array([2., 0, 0, 0, 0, 0, 0, 0, 0, 0])


def packet():
    return dict(defined=True, side_selection_defined=True, interior_is_positive=False,
                vertices=np.array([[0., 0., 0.], [1., 0., 0.], [0., 1., 0.], [0., 0., 1.]]),
                faces=[[0, 2, 1], [0, 1, 3], [0, 3, 2], [1, 2, 3]],
                face_is_cap=[0, 1, 0, 0], cap_face_count=1, surface_type="PLANE",
                vertex_velocity=np.ones((4, 3, 10)))


def backend(payload):
    calls = []

    def source(q, design, triangulation):
        calls.append("motion")
        return payload

    def audit(vertices, faces, rule):
        calls.append("physical_audit")
        assert rule == "AnyNonzero"
        return dict(boundary_valid=True, input_digest="fixture")

    native = SimpleNamespace(evaluate_spin_it_faster_source_motion=source,
                             evaluate_spin_it_faster_stratum_source_motion=source,
                             audit_exact_triangle_boundary=audit)
    return SimpleNamespace(native=native, target_design=np.eye(3),
                           triangulation=SimpleNamespace(algebraic_finite_map=True),
                           request=SimpleNamespace(target_axes=np.eye(3), target_origin=np.zeros(3)),
                           length=1., normalization_origin=np.zeros(3)), calls


def test_default_motion_and_value_fallback_keep_exact_geometry_and_audit():
    be, calls = backend(packet())
    joint = native_operand(be, Q)
    value = native_operand(be, Q, with_jacobian=False)
    assert calls == ["motion", "physical_audit"] * 2
    for field in ("vertices", "faces", "face_is_cap"):
        np.testing.assert_array_equal(getattr(value, field), getattr(joint, field))
    assert value.operation == joint.operation == "difference"
    assert value.vertex_velocity is None
    assert joint.vertex_velocity.shape == (4, 3, 10)
    assert native_operand_capabilities() == dict(value_source_available=False,
        moment_only_source_mode="motion_fallback", source_motion_avoided=False)


def test_explicit_typed_value_entry_avoids_motion_at_derivative_seam():
    data = packet()
    del data["vertex_velocity"]
    data["source_map_version"] = "ALGEBRAIC_FINITE_GRAPH_EXTRUSION_R3_R8_V1"
    be, calls = backend(data)
    chart = RawChart("PLANE", np.eye(10)[:, 1:4])

    def value(q, design, settings):
        calls.append("typed_value")
        return data

    def moving(*args):
        raise ValueError("native derivative seam")

    extension = SimpleNamespace(source_value=value, source_motion=moving)
    result = native_operand(be, Q, chart=chart, source_backend=extension, with_jacobian=False)
    assert result.vertex_velocity is None
    assert calls == ["typed_value", "physical_audit"]
    assert native_operand_capabilities(chart=chart, source_backend=extension)["source_motion_avoided"]
    with pytest.raises(ValueError, match="derivative seam"):
        native_operand(be, Q, chart=chart, source_backend=extension)
    with pytest.raises(ValueError, match="typed"):
        native_operand(be, Q, source_backend=extension, with_jacobian=False)
    with pytest.raises(ValueError, match="typed"):
        operand_from_value_source(data, Q)


def test_value_mode_never_drops_source_side_or_physical_boundary_rejection():
    data = packet()
    data["side_selection_defined"] = False
    be, calls = backend(data)
    with pytest.raises(ValueError, match="selector"):
        native_operand(be, Q, with_jacobian=False)
    assert calls == ["motion"]
    data["side_selection_defined"] = True
    be.native.audit_exact_triangle_boundary = lambda *args: dict(boundary_valid=False, input_digest="bad")
    with pytest.raises(ValueError, match="physical source"):
        native_operand(be, Q, with_jacobian=False)
    calls.clear()
    with pytest.raises(ValueError, match="normalized"):
        native_operand(be, np.zeros(10), with_jacobian=False)
    assert calls == []


def test_typed_value_has_no_derivative_basis_requirement_but_checks_surface():
    data = packet()
    data.pop("vertex_velocity")
    # No coefficient_basis: value availability does not certify a smooth chart.
    chart = RawChart("PLANE", np.eye(10)[:, 1:4])
    assert operand_from_value_source(data, Q, chart=chart).vertex_velocity is None
    with pytest.raises(ValueError, match="surface"):
        operand_from_value_source(data, Q, chart=RawChart("PARABOLOID", np.eye(10)[:, 1:9]))
