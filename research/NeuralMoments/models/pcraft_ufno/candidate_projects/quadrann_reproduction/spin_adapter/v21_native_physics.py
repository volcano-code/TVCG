"""Native label service for independent comparison with a learned finite-domain operator.

Use v21_native.native_operand for conditioning. This module performs the actual
Boolean and integration; its outputs belong to numerical evidence and supervision.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

if __package__:
    from .v21_native import FiniteOperand, RawChart, native_operand, unit_to_raw_jacobian
else:
    from v21_native import FiniteOperand, RawChart, native_operand, unit_to_raw_jacobian


_NATIVE_ERRORS = (ValueError, RuntimeError, ArithmeticError)


@dataclass
class BoundaryWitness:
    """Actual retained target-frame boundary, with motion in this call's parameter basis."""

    vertices: np.ndarray
    faces: np.ndarray
    vertex_velocity: np.ndarray | None = None


@dataclass
class NativePhysicalTargets:
    """Numerical validity and independently unknown neighbourhood events are distinct."""

    moments: np.ndarray | None = None
    editable_moments: np.ndarray | None = None
    full_jacobian: np.ndarray | None = None
    restricted_jacobian: np.ndarray | None = None
    chart: RawChart | None = None
    boundary_witness: BoundaryWitness | None = None
    moment_diagnostic: str | None = None
    derivative_diagnostic: str | None = None
    evaluation_map_identity: str | None = None
    geometry_hash: str | None = None
    topology_hash: str | None = None
    ambient_event_valid: int = -1
    restricted_event_valid: int = -1

    @property
    def moment_valid(self):
        return self.moments is not None

    @property
    def full_j_valid(self):
        return self.full_jacobian is not None

    @property
    def restricted_valid(self):
        return self.restricted_jacobian is not None


def targets_from_full_packet(packet, query):
    """Decode a successful full production call, including a valid Moment-only result."""
    result = NativePhysicalTargets(evaluation_map_identity=packet.get('evaluation_map_identity'),
                                   geometry_hash=packet.get('geometry_hash'),topology_hash=packet.get('topology_hash'))
    m = np.asarray(packet['raw_moments'], dtype=np.float64)
    edit = np.asarray(packet['variable_raw_moments'], dtype=np.float64)
    if m.shape != (10,) or edit.shape != (10,) or not np.isfinite(m).all() or not np.isfinite(edit).all():
        result.moment_diagnostic = 'invalid native ten-moment payload'
        return result
    result.moments = m.copy()
    result.editable_moments = edit.copy()
    if not packet.get('derivative_defined'):
        result.derivative_diagnostic = str(packet.get('derivative_diagnostic','native full derivative unavailable'))
        return result
    unit_j = np.asarray(packet['moment_jacobian'], dtype=np.float64)
    if unit_j.shape != (10,10) or not np.isfinite(unit_j).all():
        result.derivative_diagnostic = 'invalid native unit-coefficient derivative'
        return result
    raw_j = unit_j @ unit_to_raw_jacobian(query)
    if np.isfinite(raw_j).all():
        result.full_jacobian = raw_j
    else:
        result.derivative_diagnostic = 'invalid raw-coefficient pullback'
    return result


def _construction_records(evaluator, raw_records, physical_source, exact_fraction):
    lookup = evaluator._fixed_operands.with_source(physical_source)
    records = []
    for row in raw_records:
        operands, coordinates = [], []
        for tokens in row['operand_coordinates_exact']:
            coord = tuple(exact_fraction(x) for x in tokens)
            candidates = lookup(coord)
            if len(candidates) != 1:
                raise ValueError('Boolean moving-source correspondence unavailable')
            operands.append(dict(source_vertices=candidates, status='UNIQUE'))
            coordinates.append(list(map(float, coord)))
        records.append(dict(vertex_index=row['vertex_index'],point_type=row['point_type'],
                            parameter_t=row['parameter_t'],operands=operands,operand_coordinates=coordinates))
    return records


def evaluate_operand_physics(evaluator, operand: FiniteOperand, *, chart: RawChart | None = None,
                             capture_boundary: bool = False):
    """Integrate an explicit finite operand, using only the supplied raw motion columns.

    For a frozen operand, omit velocity to obtain Moment alone. On a typed
    stratum the Jacobian is 10 by the chart dimension and stays in that chart.
    """
    from make_it_stand.verification import cherchi_inprocess as ci
    from make_it_stand.verification.cherchi_motion_differential import recover_construction_vertex_motion

    result = NativePhysicalTargets(chart=chart, evaluation_map_identity=evaluator.context_identity)
    request = evaluator.request
    axes = np.asarray(request.target_axes)
    physical = ((operand.vertices @ axes.T + request.target_origin) * evaluator.length +
                evaluator.normalization_origin)
    try:
        packet = evaluator._boolean_with_valid_source(physical,operand.faces,operand.operation)
        evaluator._validate_packet(packet)
        exact = tuple(tuple(ci.exact_fraction(x) for x in row) for row in packet['exact_vertices'])
        retained = np.asarray(packet['vertices'])
        faces, _ = ci.normalize_closed_component_orientation(retained,np.asarray(packet['faces']),rational_vertices=exact)
        target_vertices, rational = evaluator._target_view(exact)
        tokens = [[f'{x.numerator}/{x.denominator}' for x in row] for row in rational]
        boundary = evaluator.arithmetic.PreparedBoundary(tokens,np.asarray(faces,dtype=np.int64))
        editable = np.asarray(evaluator._integrate(target_vertices,faces,rational,prepared_boundary=boundary))
        if editable.shape != (10,) or not np.isfinite(editable).all():
            raise ValueError('invalid integrated finite-operand moments')
        result.editable_moments = editable
        result.moments = editable + evaluator.fixed
        result.geometry_hash = ci.digest(dict(vertices=packet['exact_vertices'],faces=faces.tolist()))
        if capture_boundary:
            result.boundary_witness = BoundaryWitness(target_vertices,faces)
    except _NATIVE_ERRORS as exc:
        result.moment_diagnostic = str(exc)
        return result
    velocity = operand.vertex_velocity
    if velocity is None:
        result.derivative_diagnostic = 'frozen finite operand has no supplied motion'
        return result
    dim = 10 if chart is None else chart.basis.shape[1]
    if velocity.shape[-1] != dim:
        result.derivative_diagnostic = 'velocity dimension differs from requested raw coordinate chart'
        return result
    try:
        records = _construction_records(evaluator,packet['construction_records'],physical,ci.exact_fraction)
        physical_velocity = np.einsum('vkd,jk->vjd',velocity,axes)*evaluator.length
        moving = recover_construction_vertex_motion(records,evaluator.design,physical,physical_velocity,retained)
        target_motion = np.einsum('vkd,kj->vjd',moving/evaluator.length,axes)
        if capture_boundary:
            result.boundary_witness.vertex_velocity = target_motion
        jac = np.asarray(evaluator._differentiate(target_vertices,faces,target_motion,rational,prepared_boundary=boundary))
        if jac.shape != (10,dim) or not np.isfinite(jac).all():
            raise ValueError('invalid moving finite-operand differential')
        if chart is None:
            result.full_jacobian = jac
        else:
            result.restricted_jacobian = jac
    except _NATIVE_ERRORS as exc:
        result.derivative_diagnostic = str(exc)
    return result


def evaluate_native_physics(evaluator, query, *, chart: RawChart | None = None):
    """Call the configured deployed map; rejected labels carry no fabricated values."""
    try:
        if chart is not None:
            operand = native_operand(evaluator,query,chart=chart)
            return evaluate_operand_physics(evaluator,operand,chart=chart)
        try:
            return targets_from_full_packet(evaluator(query),query)
        finally:
            evaluator._materials.clear()
    except _NATIVE_ERRORS as exc:
        return NativePhysicalTargets(chart=chart,evaluation_map_identity=evaluator.context_identity,
                                     moment_diagnostic=str(exc))
