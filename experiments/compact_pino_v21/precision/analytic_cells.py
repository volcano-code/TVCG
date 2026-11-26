"""One-cell analytic counterexample for full-cell response cubature.

All rules reproduce the ten full-cell moments. That does not make their
integrals of moment basis times a cubic retention exact. An oblique planar
face, placed far from the other faces of a closed finite operand, isolates
this difference on the actual evaluate_packed path. The complete cell lies
strictly inside the cubic transition, so independent monomial integration is
an exact smooth-response reference in real arithmetic. It is not a native
hard-Boolean reference, a learned-network test, or solver acceptance.
"""
from __future__ import annotations

import argparse
from importlib import import_module
import json
from pathlib import Path
import tempfile
import time

import numpy as np

from ..p0_diagnostics.baseline_cases import PREFIX, jsonable


RULES = ("tetra4", "gl2", "gl3")
POWERS = ((0, 0, 0), (1, 0, 0), (0, 1, 0), (0, 0, 1), (2, 0, 0),
          (0, 2, 0), (0, 0, 2), (0, 1, 1), (1, 0, 1), (1, 1, 0))
SCHEMA = "tvcg.v22.analytic-cell-cubature.v1"


def _multiply(left, right):
    result = {}
    for a, x in left.items():
        for b, y in right.items():
            power = tuple(i + j for i, j in zip(a, b))
            result[power] = result.get(power, 0.) + x * y
    return result


def _integral(polynomial, lo, hi, extra=(0, 0, 0)):
    """Independent rectangular monomial integral; no quadrature nodes used."""
    value = 0.
    for powers, coefficient in polynomial.items():
        degree = np.asarray(powers) + np.asarray(extra) + 1
        value += coefficient * np.prod((hi ** degree - lo ** degree) / degree)
    return value


def _affine(constant, vector):
    return {(0, 0, 0): float(constant), (1, 0, 0): float(vector[0]),
            (0, 1, 0): float(vector[1]), (0, 0, 1): float(vector[2])}


def plane_frame(normal):
    normal = np.asarray(normal, dtype=np.float64)
    if normal.shape != (3,) or not np.isfinite(normal).all() or not np.linalg.norm(normal):
        raise ValueError("a finite nonzero three-dimensional plane normal is required")
    normal = normal / np.linalg.norm(normal)
    axis = np.eye(3)[np.argmin(np.abs(normal))]
    tangent_u = np.cross(axis, normal)
    tangent_u /= np.linalg.norm(tangent_u)
    tangent_v = np.cross(normal, tangent_u)
    return np.column_stack((tangent_u, tangent_v, normal))


def polynomial_reference(lo, hi, *, normal, offset, bandwidth, density=1.):
    """Exact cubic-response M and J for offset plus two local plane tilts.

    The inside convention is n.x <= offset. At zero tilts the signed-distance
    motions are [1, u.x, v.x]. The plane frame is orthonormal, so these are
    derivatives of Euclidean signed distance, not of a vertical-gap surrogate.
    """
    lo, hi = np.asarray(lo, dtype=float), np.asarray(hi, dtype=float)
    if (lo.shape != (3,) or hi.shape != (3,) or not np.isfinite([lo, hi]).all()
            or np.any(hi <= lo) or not np.isfinite([offset, bandwidth, density]).all()
            or bandwidth <= 0 or density <= 0):
        raise ValueError("finite positive cell spans, bandwidth and density are required")
    frame = plane_frame(normal)
    n = frame[:, 2]
    corners = np.array(np.meshgrid(*zip(lo, hi), indexing="ij")).reshape(3, -1).T
    signed = (offset - corners @ n) / bandwidth
    if np.max(np.abs(signed)) >= 1:
        raise ValueError("the whole cell must lie strictly inside the cubic transition")
    s = _affine(offset / bandwidth, -n / bandwidth)
    squared = _multiply(s, s)
    cubed = _multiply(squared, s)
    retention = {(0, 0, 0): .5}
    for polynomial, factor in ((s, .75), (cubed, -.25)):
        for powers, value in polynomial.items():
            retention[powers] = retention.get(powers, 0.) + factor * value
    slope = {powers: -.75 * value / bandwidth for powers, value in squared.items()}
    slope[(0, 0, 0)] = slope.get((0, 0, 0), 0.) + .75 / bandwidth
    motions = ({(0, 0, 0): 1.}, _affine(0., frame[:, 0]), _affine(0., frame[:, 1]))
    derivatives = [_multiply(slope, motion) for motion in motions]
    moments = density * np.array([_integral(retention, lo, hi, power) for power in POWERS])
    jacobian = density * np.array([[_integral(poly, lo, hi, power) for poly in derivatives]
                                  for power in POWERS])
    full = density * np.array([_integral({(0, 0, 0): 1.}, lo, hi, power) for power in POWERS])
    return dict(moments=moments, jacobian=jacobian, full_moments=full,
                frame=frame, maximum_abs_signed_coordinate=float(np.max(np.abs(signed))))


def _finite_plane(frame, offset, extent):
    """Closed distant box with a moving oblique top face, in a proper frame."""
    local = np.array([[-extent, -extent, -extent], [extent, -extent, -extent],
                      [extent, extent, -extent], [-extent, extent, -extent],
                      [-extent, -extent, offset], [extent, -extent, offset],
                      [extent, extent, offset], [-extent, extent, offset]])
    faces = np.array([[0, 2, 1], [0, 3, 2], [4, 5, 6], [4, 6, 7],
                      [0, 1, 5], [0, 5, 4], [1, 2, 6], [1, 6, 5],
                      [2, 3, 7], [2, 7, 6], [3, 0, 4], [3, 4, 7]])
    vertices = local @ frame.T
    motion = np.zeros((8, 3, 3))
    motion[4:] = frame[:, 2][None, :, None] * np.column_stack((np.ones(4), local[4:, :2]))[:, None, :]
    return vertices, faces, motion


def slanted_cell_report(cache_directory, *, pitch=1., density=1.3, offset_fraction=.07,
                        normal=(1., 2., 3.), rules=RULES, chunk_size=7):
    """Run finite-operand production reductions against independent polynomials."""
    if (not np.isfinite([pitch, density, offset_fraction]).all() or pitch <= 0 or density <= 0
            or type(chunk_size) is not int or chunk_size <= 0):
        raise ValueError("positive pitch, density and integer chunk_size are required")
    if not rules or len(set(rules)) != len(rules) or any(rule not in RULES for rule in rules):
        raise ValueError("rules must be a nonempty unique selection of tetra4, gl2, gl3")
    material_module = import_module(PREFIX + ".v21_material")
    blocks_type = import_module(PREFIX + ".v21_blocks").BlockMeasure
    response_module = import_module(PREFIX + ".v21_response")
    evaluate_packed = import_module(PREFIX + ".compact_pino_v21").evaluate_packed
    bandwidth, offset = pitch, offset_fraction * pitch
    lo, hi = np.full(3, -.5 * pitch), np.full(3, .5 * pitch)
    reference = polynomial_reference(lo, hi, normal=normal, offset=offset,
                                     bandwidth=bandwidth, density=density)
    # One full native-grid cell centred at the origin; no boundary cubature.
    origin = np.full(3, -128.5 * pitch)
    material = material_module.prepare_rle_material(np.array([[128, 128, 128, 129]], dtype=np.int32),
                                                     origin, pitch, side=256, density=density)
    directory = Path(cache_directory) / (material.identity + "-cell")
    blocks = blocks_type.build(material, directory, block_voxels=1)
    vertices, faces, motion = _finite_plane(reference["frame"], offset, 10. * pitch)
    surface = response_module.prepare_surface(vertices, faces)
    scale = density * pitch ** (3 + np.array([sum(p) for p in POWERS]))
    parameter_scales = np.array([pitch, 1., 1.])
    mask = np.ones(len(blocks.ids), dtype=bool)
    rows = []
    for rule in rules:
        full = np.zeros(10)
        node_count, minimum_weight = 0, float("inf")
        for nodes, weights in blocks.chunks(mask, max_nodes=chunk_size, full_cell_rule=rule):
            monomials = np.array([np.prod(nodes ** np.asarray(p), axis=1) for p in POWERS]).T
            full += monomials.T @ weights
            node_count += len(weights)
            minimum_weight = min(minimum_weight, float(np.min(weights)))
        started = time.perf_counter()
        response = evaluate_packed(blocks, surface, bandwidth, "intersection", np.zeros(10), motion,
                                   model=None, chunk_size=chunk_size, full_cell_rule=rule)
        seconds = time.perf_counter() - started
        opposite = evaluate_packed(blocks, surface, bandwidth, "difference", np.zeros(10), motion,
                                   model=None, chunk_size=chunk_size, full_cell_rule=rule)
        values = response.moments.detach().cpu().numpy()
        jacobian = None if response.jacobian is None else response.jacobian.detach().cpu().numpy()
        if not np.isfinite(values).all() or jacobian is None or not np.isfinite(jacobian).all():
            raise ValueError("analytic regular plane unexpectedly lacks finite response or derivative")
        moment_error = values - reference["moments"]
        jacobian_error = jacobian - reference["jacobian"]
        complement = values + opposite.moments.detach().cpu().numpy() - reference["full_moments"]
        if opposite.jacobian is None:
            raise ValueError("analytic complement unexpectedly lacks a regular derivative")
        rows.append(dict(full_cell_rule=rule, actual_nodes=node_count, minimum_weight=minimum_weight,
                         full_moment_scaled_error=(full-reference["full_moments"])/scale,
                         moments=values, jacobian=jacobian, raw_moment_error=moment_error,
                         moment_error_l2=float(np.linalg.norm(moment_error)),
                         scaled_moment_error=moment_error/scale,
                         raw_jacobian_error=jacobian_error,
                         chart_unit_Jv_scaled_error=jacobian_error*parameter_scales[None, :]/scale[:, None],
                         complement_scaled_error=complement/scale,
                         complement_jacobian_max=float(np.max(np.abs(jacobian+opposite.jacobian.detach().cpu().numpy()))),
                         query_seconds=seconds, diagnostics=response.diagnostics))
    return jsonable(dict(schema=SCHEMA, status="SYNTHETIC_COUNTEREXAMPLE_NOT_NATIVE_ACCEPTANCE",
                         reference_kind="analytic_cubic_smoothed_finite_plane",
                         reference_method="independent_rectangular_monomial_antiderivatives",
                         model_weights_used=False, parameter_updates=0,
                         native_solver_acceptance="NOT_EVALUATED", original_five_percent_gate="UNCHANGED_NOT_EVALUATED",
                         fixed_grid=dict(side=256, origin=origin, pitch=pitch, h=bandwidth, h_over_pitch=1.),
                         cell_bounds=[lo, hi], density=density, offset=offset,
                         normal=reference["frame"][:, 2], frame=reference["frame"],
                         maximum_abs_signed_coordinate=reference["maximum_abs_signed_coordinate"],
                         moment_order=["1", "x", "y", "z", "xx", "yy", "zz", "yz", "xz", "xy"],
                         derivative_coordinates=["normal_offset", "u_tilt", "v_tilt"],
                         parameter_scales=parameter_scales,
                         reference_moments=reference["moments"], reference_jacobian=reference["jacobian"],
                         full_cell_moments=reference["full_moments"], material_identity=material.identity,
                         material_source=material.source, material_partial_cells=material.partial_cell_count,
                         timing_scope="one evaluate_packed call; excludes preparation, polynomial reference and complement check",
                         interpretation="full-cell ten-moment exactness alone does not imply cubic-response M/J exactness; gl3 exactness is specific to this unclipped planar polynomial fixture",
                         records=rows))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    parser.add_argument("--rules", nargs="+", choices=RULES, default=list(RULES))
    parser.add_argument("--pitch", type=float, default=1.)
    parser.add_argument("--density", type=float, default=1.3)
    parser.add_argument("--chunk-size", type=int, default=7)
    args = parser.parse_args(argv)
    output = Path(args.output)
    if output.exists():
        parser.error("output must be new; existing diagnostic reports are never overwritten")
    with tempfile.TemporaryDirectory(prefix="precision-cell-") as cache:
        result = slanted_cell_report(cache, pitch=args.pitch, density=args.density,
                                     rules=args.rules, chunk_size=args.chunk_size)
    with output.open("x", encoding="utf8") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps(dict(output=str(output.resolve()), status=result["status"],
                          native_solver_acceptance="NOT_EVALUATED")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
