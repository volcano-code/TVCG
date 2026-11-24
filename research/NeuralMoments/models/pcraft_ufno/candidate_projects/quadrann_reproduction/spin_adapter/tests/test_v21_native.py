"""Fail if a native operand loses its operation, cap membership or raw-coordinate motion."""
from dataclasses import fields
from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from v21_native import FiniteOperand, RawChart, operand_from_source


Q = np.array([2., 0., 0., 0., 0., 0., 0., 0., 0., 0.])


def source():
    vertices = np.array([[0., 0., 0.], [1., 0., 0.], [0., 1., 0.], [0., 0., 1.]])
    velocity = np.broadcast_to(np.arange(10.), (*vertices.shape, 10)).copy()
    velocity[..., 0] = 9.
    return dict(defined=True, side_selection_defined=True, interior_is_positive=False,
                vertices=vertices, faces=[[0, 2, 1], [0, 1, 3], [0, 3, 2], [1, 2, 3]],
                face_is_cap=[0, 1, 0, 0], cap_face_count=1, vertex_velocity=velocity)


def test_raw_nonunit_pullback_occurs_once_and_uses_cap_ids(tmp_path):
    operand = operand_from_source(source(), Q)
    assert operand.operation == "difference"
    assert operand.face_is_cap.tolist() == [False, True, False, False]
    np.testing.assert_array_equal(operand.vertex_velocity[0, 0], [0, .5, 1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5])
    path = tmp_path / "operand.npz"
    operand.save(path)
    readback = FiniteOperand.load(path)
    assert readback.operation == operand.operation
    for key in ["vertices", "faces", "face_is_cap", "vertex_velocity"]:
        np.testing.assert_array_equal(getattr(readback, key), getattr(operand, key))
    assert {f.name for f in fields(operand)} == {"vertices", "faces", "face_is_cap", "operation", "vertex_velocity"}
    stationary = FiniteOperand(operand.vertices, operand.faces, operand.face_is_cap, "intersection")
    stationary.save(path)
    assert FiniteOperand.load(path).vertex_velocity is None


@pytest.mark.parametrize("dimension,surface", [(3, "PLANE"), (8, "PARABOLOID")])
def test_raw_reduced_velocity_only_in_native_tangent_coordinates(dimension, surface):
    native_basis = np.eye(10)[:, 1:dimension+1]
    raw_basis = native_basis[:, ::-1].copy()
    raw_basis[0] = 2  # project away the radial component before matching frames
    chart = RawChart(surface, raw_basis)
    packet = source()
    packet.update(surface_type=surface, coefficient_basis=native_basis,
                  vertex_velocity=packet["vertex_velocity"][..., :dimension])
    operand = operand_from_source(packet, Q, chart=chart)
    expected = packet["vertex_velocity"][0, 0, ::-1] / 2
    np.testing.assert_array_equal(operand.vertex_velocity[0, 0], expected)
    assert operand.vertex_velocity.shape == (4, 3, dimension)
    raw_basis[-1, 0] = 1
    with pytest.raises(ValueError, match="tangent"):
        operand_from_source(packet, Q, chart=RawChart(surface, raw_basis))


def test_undefined_side_and_mislabeled_cap_never_create_operands():
    packet = source()
    packet["side_selection_defined"] = False
    with pytest.raises(ValueError, match="selector|side"):
        operand_from_source(packet, Q)
    packet = source()
    packet["cap_face_count"] = 2
    with pytest.raises(ValueError, match="cap"):
        operand_from_source(packet, Q)
    packet = source()
    packet["faces"][0][-1] = 100
    with pytest.raises(ValueError, match="faces"):
        operand_from_source(packet, Q)


def test_physics_moment_only_is_retained_and_unavailable_derivatives_remain_absent():
    from v21_native_physics import targets_from_full_packet
    packet = dict(raw_moments=np.arange(10.), variable_raw_moments=np.arange(10.) - 1.,
                  derivative_defined=False, moment_jacobian=np.zeros((10, 10)),
                  derivative_diagnostic="unavailable")
    result = targets_from_full_packet(packet, Q)
    assert result.moment_valid and not result.full_j_valid and not result.restricted_valid
    np.testing.assert_array_equal(result.moments, np.arange(10.))
    assert result.full_jacobian is None and result.restricted_jacobian is None
    assert result.ambient_event_valid == -1 and result.restricted_event_valid == -1


def test_physics_raw_derivative_scaling_and_rejects_nonfinite_payload():
    from v21_native_physics import targets_from_full_packet
    packet = dict(raw_moments=np.arange(10.), variable_raw_moments=np.arange(10.),
                  derivative_defined=True, moment_jacobian=np.eye(10))
    result = targets_from_full_packet(packet, Q)
    np.testing.assert_array_equal(result.full_jacobian, np.diag([0.] + [.5] * 9))
    assert result.restricted_jacobian is None
    packet['moment_jacobian'][0, 1] = np.nan
    bad = targets_from_full_packet(packet, Q)
    assert bad.moment_valid and not bad.full_j_valid and bad.derivative_diagnostic
