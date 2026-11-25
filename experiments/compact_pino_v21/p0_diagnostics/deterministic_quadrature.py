"""Independent synthetic plane/parabolic graph integration and V2.1 band sweep.

The graph fixtures are not the deployed finite operand. They isolate static
cubature and smoothing error with analytic references. Real finite-operand
sweeps call the actual V2.1 evaluate_packed implementation, model=None.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import time
import numpy as np
from numpy.polynomial.legendre import leggauss
try:
    from .baseline_cases import (moment_scale, finite_array, jsonable, load_factory,
                                 load_baseline_case, validate_case, operand_fingerprint,
                                 write_synthetic_finite_case)
except ImportError:  # Direct script execution from this directory.
    from baseline_cases import (moment_scale, finite_array, jsonable, load_factory,
                                load_baseline_case, validate_case, operand_fingerprint,
                                write_synthetic_finite_case)


def basis(xyz):
    x, y, z = xyz.T
    return np.column_stack((np.ones(len(x)), x, y, z, x*x, y*y, z*z, y*z, x*z, x*y))


def boxes_checked(boxes):
    b = np.asarray(boxes, dtype=np.float64)
    if b.ndim != 3 or b.shape[1:] != (2, 3) or len(b)==0 or not np.isfinite(b).all() or (b[:,1] <= b[:,0]).any():
        raise ValueError("nonempty finite boxes [N,2,3] with positive spans required")
    for i, a in enumerate(b):
        if i and np.any(np.all(np.minimum(a[1], b[:i,1]) > np.maximum(a[0], b[:i,0]), axis=1)):
            raise ValueError("material boxes must have disjoint interiors")
    return b


def full_moments(boxes):
    """Independent analytic moments of disjoint unit-density boxes."""
    b = boxes_checked(boxes); lo, hi = b[:,0], b[:,1]
    i0=hi-lo; i1=(hi**2-lo**2)/2; i2=(hi**3-lo**3)/3
    x,y,z=i0.T; a,c,d=i1.T; aa,cc,dd=i2.T
    return np.array([x*y*z,a*y*z,x*c*z,x*y*d,aa*y*z,x*cc*z,x*y*dd,x*c*d,a*y*d,a*c*z]).sum(1)


def graph_reference(boxes, offset, curvature=0.):
    """FP64 polynomial integration of z <= offset + curvature*x^2.

    Split at all x where the graph meets a horizontal material face. Four-point
    Gauss integrates the resulting polynomials exactly in real arithmetic.
    J columns are d/d offset and d/d curvature. Reject a planar face event.
    """
    b=boxes_checked(boxes)
    if not np.isfinite([offset, curvature]).all(): raise ValueError("finite graph required")
    gx,gw=leggauss(4); M=np.zeros(10); J=np.zeros((10,2)); regular=True
    for (xl,yl,zl),(xh,yh,zh) in b:
        Y=yh-yl; Y1=(yh*yh-yl*yl)/2; Y2=(yh**3-yl**3)/3
        breaks=[xl,xh]
        if curvature == 0:
            if offset == zl or offset == zh: regular=False
        else:
            for z in (zl,zh):
                r=(z-offset)/curvature
                if r>=0:
                    for x in (-np.sqrt(r),np.sqrt(r)):
                        if xl<x<xh: breaks.append(x)
        breaks=sorted(set(breaks))
        for a,c in zip(breaks[:-1],breaks[1:]):
            x=(a+c)/2+(c-a)/2*gx; w=(c-a)/2*gw
            graph=offset+curvature*x*x; z=np.clip(graph,zl,zh)
            Z=z-zl; Z1=(z*z-zl*zl)/2; Z2=(z**3-zl**3)/3
            f=np.column_stack((Y*Z,x*Y*Z,Y1*Z,Y*Z1,x*x*Y*Z,Y2*Z,Y*Z2,Y1*Z1,x*Y*Z1,x*Y1*Z))
            M+=w@f
            active=(graph>zl)&(graph<zh)
            # Reynolds/Leibniz boundary derivative, no derivative through a hard atom mask.
            g=np.column_stack((np.full(len(x),Y),x*Y,np.full(len(x),Y1),Y*z,x*x*Y,np.full(len(x),Y2),Y*z*z,Y1*z,x*Y*z,x*Y1))
            J[:,0]+=(w*active)@g; J[:,1]+=(w*active*x*x)@g
    return M, J if regular else None


def static_atoms(boxes, pitch, order=2, max_nodes=2_000_000):
    """Positive tensor Gauss atoms on subcells; exact degree-two box moments."""
    if not np.isfinite(pitch) or pitch <= 0 or type(order) is not int or order<2:
        raise ValueError("positive pitch and Gauss order >=2 required")
    nodes=[]; weights=[]; total=0; g,w=leggauss(order)
    for lo,hi in boxes_checked(boxes):
        counts=np.maximum(1,np.ceil((hi-lo)/pitch-1e-12).astype(int))
        total+=int(np.prod(counts))*order**3
        if total>max_nodes: raise ValueError("synthetic atom budget exceeded")
        xyz=[]; ww=[]
        for l,h,n in zip(lo,hi,counts):
            step=(h-l)/n; centers=l+(np.arange(n)+.5)*step
            xyz.append((centers[:,None]+.5*step*g[None]).ravel())
            ww.append(np.tile(w*.5*step,n))
        nodes.append(np.stack(np.meshgrid(*xyz,indexing="ij"),axis=-1).reshape(-1,3))
        weights.append(np.prod(np.stack(np.meshgrid(*ww,indexing="ij"),axis=-1),axis=-1).ravel())
    return np.concatenate(nodes),np.concatenate(weights)


def smooth_graph(nodes, weights, offset, curvature, bandwidth, operation="intersection"):
    """C1 cubic vertical-gap surrogate. For curved graphs NOT Euclidean SDF.

    Same-operand difference complements retention; all derivatives are of
    this same approximate response, not of the hard native Boolean.
    """
    if operation not in ("intersection","difference") or not np.isfinite(bandwidth) or bandwidth<=0:
        raise ValueError("invalid operation or bandwidth")
    sign=1 if operation=="intersection" else -1
    x,_,z=nodes.T; s=sign*(offset+curvature*x*x-z)/bandwidth
    t=(np.clip(s,-1,1)+1)/2; r=t*t*(3-2*t)
    slope=np.where(np.abs(s)<1,sign*.75*(1-s*s)/bandwidth,0.)
    W=basis(nodes)*weights[:,None]
    return W.T@r, W.T@np.column_stack((slope,slope*x*x))


def synthetic_suite():
    # All fixtures are disjoint unit-density boxes, not production assets.
    fixtures={"long_prism":[[[-.5,-.1,-.8],[.5,.1,.8]]],
              "thin_wall":[[[-.5,-.01,-.5],[.5,.01,.5]]],
              "disconnected":[[[-.5,-.4,-.5],[-.1,.4,.5]],[[.1,-.4,-.5],[.5,.4,.5]]],
              "cavity":[[[-.5,-.5,-.5],[-.2,.5,.5]],[[.2,-.5,-.5],[.5,.5,.5]],
                        [[-.2,-.5,-.5],[.2,-.2,.5]],[[-.2,.2,-.5],[.2,.5,.5]],
                        [[-.2,-.2,-.5],[.2,.2,-.2]],[[-.2,-.2,.2],[.2,.2,.5]]]}
    records=[]
    for name,boxes in fixtures.items():
        full=full_moments(boxes); b=boxes_checked(boxes); L=float(np.max(b[:,1].max(0)-b[:,0].min(0)))
        scale=moment_scale(full[0],L); pitch=L/24; nodes,w=static_atoms(boxes,pitch)
        full_err=float(np.max(np.abs(basis(nodes).T@w-full)/scale))
        for kind,a in (("plane",0.),("parabolic_graph",.4)):
            t=.037; ref,jref=graph_reference(boxes,t,a); assert jref is not None
            for k in (1,2,4):
                start=time.perf_counter(); m,j=smooth_graph(nodes,w,t,a,k*pitch); elapsed=time.perf_counter()-start
                mr,jr=smooth_graph(nodes,w,t,a,k*pitch,"difference")
                eps=1e-5*pitch
                mp,_=smooth_graph(nodes,w,t+eps,a,k*pitch); mm,_=smooth_graph(nodes,w,t-eps,a,k*pitch)
                fd=(mp-mm)/(2*eps)
                records.append(dict(geometry=name,cut=kind,bandwidth_voxels=k,nodes=len(w),
                                    full_material_scaled_max=full_err,moment_scaled_max=float(np.max(np.abs(m-ref)/scale)),
                                    offset_derivative_scaled_max=float(np.max(np.abs(j[:,0]-jref[:,0])/scale)*L),
                                    curvature_derivative_scaled_max=float(np.max(np.abs(j[:,1]-jref[:,1])/scale)/L),
                                    fd_offset_scaled_max=float(np.max(np.abs(fd-j[:,0])/scale)*L),
                                    complement_scaled_max=float(np.max(np.abs(m+mr-full)/scale)),
                                    complement_J_max=float(np.max(np.abs(j+jr))),seconds=elapsed))
    return dict(schema="tvcg.v21.synthetic-quadrature.v1",scope="synthetic_only_graph_reference_not_native_finite_operand",records=records)


def _audit_blocks(blocks, scale, material_cache_moments=None, chunk_size=65536):
    """Verify positive cached points and per-block aggregates before timing queries.

    This is a diagnostic integrity pass, not a production per-query operation.
    The packed cache does not itself checksum nodes.npy, so validating only its
    material-name field would miss stale/corrupt rows or block assignments.
    """
    count = len(blocks.ids)
    if not count or blocks.nodes.ndim != 2 or blocks.nodes.shape[1] != 4:
        raise ValueError("nonempty positive material blocks required")
    if (blocks.offsets.shape != (count + 1,) or blocks.offsets[0] != 0
            or blocks.offsets[-1] != len(blocks.nodes) or np.any(np.diff(blocks.offsets) <= 0)):
        raise ValueError("invalid material block offsets")
    if blocks.moments.shape != (count, 10) or not np.isfinite(blocks.moments).all():
        raise ValueError("invalid material block moments")
    width = blocks.block_voxels
    if type(width) is not int or width <= 0 or blocks.side % width:
        raise ValueError("block width must divide native256")
    n = blocks.side // width
    ids = np.asarray(blocks.ids)
    if ids.dtype.kind not in 'iu' or np.any(ids < 0) or np.any(ids >= n**3) or np.any(np.diff(ids) <= 0):
        raise ValueError("block ids must be sorted unique valid native indices")
    origin = finite_array(blocks.origin, (3,), "block origin")
    ijk = np.array(np.unravel_index(ids, (n, n, n))).T
    edge = width * blocks.pitch
    centers = origin + (ijk + .5) * edge
    if (not np.allclose(blocks.centers, centers, rtol=0, atol=1e-12 * max(1., edge))
            or not np.isclose(blocks.radius, np.sqrt(3.) * edge / 2, rtol=1e-13, atol=0)):
        raise ValueError("invalid cached block bounds")
    digest = hashlib.sha256(b"v21-p0-checked-blocks-v1")
    total = np.zeros(10)
    maximum = 0.
    minimum_weight = np.inf
    for i in range(count):
        moments = np.zeros(10)
        lo, hi = centers[i] - edge/2, centers[i] + edge/2
        tolerance = 128*np.finfo(float).eps * max(1., float(np.max(np.abs([lo, hi]))))
        for start in range(int(blocks.offsets[i]), int(blocks.offsets[i+1]), chunk_size):
            rows = np.asarray(blocks.nodes[start:min(start+chunk_size, int(blocks.offsets[i+1]))])
            if not np.isfinite(rows).all() or np.any(rows[:,3] <= 0):
                raise ValueError("cached cubature must have finite points and positive weights")
            if np.any(rows[:,:3] < lo-tolerance) or np.any(rows[:,:3] > hi+tolerance):
                raise ValueError("cached cubature points escaped their assigned spatial block")
            moments += (basis(rows[:,:3]) * rows[:,3,None]).sum(0)
            minimum_weight = min(minimum_weight, float(rows[:,3].min()))
            digest.update(np.ascontiguousarray(rows, dtype='<f8').tobytes())
        maximum = max(maximum, float(np.max(np.abs(moments-blocks.moments[i])/scale)))
        total += moments
    if maximum > 1e-10:
        raise ValueError("cached per-block moments do not match their cubature points")
    if material_cache_moments is not None:
        cached = finite_array(material_cache_moments, (10,), "material_cache_moments")
        if np.max(np.abs(total-cached)/scale) > 1e-10:
            raise ValueError("packed cubature does not reproduce the loaded material cache")
    for array in (ids, blocks.offsets, blocks.moments, origin):
        digest.update(np.ascontiguousarray(array).tobytes())
    return dict(cubature_moments=total, packed_content_sha256=digest.hexdigest(),
                nodes=len(blocks.nodes), blocks=count, minimum_weight=minimum_weight,
                maximum_block_scaled_error=maximum)


def v21_band_sweep(case, *, chunk_size=65536):
    """Run the existing unweighted finite-operand operator at 1/2/4 voxel widths.

    The same material, vertices, caps, motion and surface are reused for both
    sides. No Boolean answer or target is an input to evaluate_packed. Native
    targets only enter errors afterwards. The complement identity is
    M_inside + M_outside = 2*M_fixed + M_editable, not M_fixed + M_editable.
    Synthetic fixtures exercising this production code path remain synthetic.
    """
    from research.NeuralMoments.models.pcraft_ufno.candidate_projects.quadrann_reproduction.spin_adapter.compact_pino_v21 import evaluate_packed
    from research.NeuralMoments.models.pcraft_ufno.candidate_projects.quadrann_reproduction.spin_adapter.v21_response import prepare_surface
    if type(chunk_size) is not int or chunk_size <= 0:
        raise ValueError("positive integer chunk_size required")
    scale = validate_case(case)
    meta, blocks, operand = case['meta'], case['blocks'], case['operand']
    fixed = finite_array(case['fixed_moments'], (10,), 'fixed_moments')
    native = finite_array(case['native_moments'], (10,), 'native_moments')
    reference_full = finite_array(case['native_full_editable_moments'], (10,), 'native_full_editable_moments')
    start = time.perf_counter()
    audit = _audit_blocks(blocks, scale, case.get('material_cache_moments'), chunk_size)
    audit_seconds = time.perf_counter() - start
    full_error = (audit['cubature_moments'] - reference_full) / scale
    audit.update(native_reference_scaled_errors=full_error,
                 maximum_native_reference_scaled_error=float(np.max(np.abs(full_error))),
                 native_reference_moment_status='PASS' if np.max(np.abs(full_error)) <= 1e-8 else 'FAIL')
    if audit['native_reference_moment_status'] != 'PASS':
        return jsonable(dict(schema='tvcg.v21.native-band-sweep.v2',
                             status='STOP_FULL_MATERIAL_REFERENCE',
                             scope='material_check_only_no_query_comparisons',
                             reason='full editable ten-moment error exceeds frozen physical-scale tolerance 1e-8',
                             native_solver_acceptance='NOT_EVALUATED', model_weights_used=False,
                             provenance=meta, asset_receipts=case.get('asset_receipts'),
                             manifest_receipt=case.get('manifest_receipt'), material_audit=audit,
                             load_and_pack_seconds=case.get('load_and_pack_seconds'),
                             material_audit_seconds=audit_seconds, records=[]))
    qualified = meta['derivative_valid']
    dim = meta['derivative_dimension']
    velocity = operand.vertex_velocity if qualified else None
    native_j = finite_array(case['native_jacobian'], (10, dim), 'native_jacobian') if qualified else None
    start = time.perf_counter()
    surface = prepare_surface(operand.vertices, operand.faces)
    setup_seconds = time.perf_counter() - start
    records = []
    primary = operand.operation
    opposite = 'difference' if primary == 'intersection' else 'intersection'
    for width in (1, 2, 4):
        responses = {}
        for operation in (primary, opposite):
            start = time.perf_counter()
            prediction = evaluate_packed(blocks, surface, width*blocks.pitch, operation,
                                         fixed, velocity, model=None, chunk_size=chunk_size)
            elapsed = time.perf_counter() - start
            moments = finite_array(prediction.moments.detach().cpu().numpy(), (10,), 'response moments')
            jacobian = (None if prediction.jacobian is None else
                        finite_array(prediction.jacobian.detach().cpu().numpy(), (10, dim), 'response Jacobian'))
            responses[operation] = dict(moments=moments, jacobian=jacobian,
                                        seconds=elapsed, diagnostics=prediction.diagnostics)
        current, other = responses[primary], responses[opposite]
        j, opposite_j = current['jacobian'], other['jacobian']
        valid = qualified and j is not None and opposite_j is not None
        status = ('AVAILABLE' if valid else 'UNAVAILABLE_NATIVE_REFERENCE'
                  if not qualified else 'UNAVAILABLE_ACTIVE_DISTANCE_NONREGULAR')
        complement = (current['moments'] + other['moments'] - 2*fixed - audit['cubature_moments'])/scale
        error = (current['moments'] - native)/scale
        item = dict(bandwidth_voxels=width, bandwidth=width*blocks.pitch,
                    operation=primary, moments=current['moments'], jacobian=j,
                    seconds=current['seconds'], diagnostics=current['diagnostics'],
                    opposite_side=other, scaled_moment_error=error,
                    maximum_scaled_moment_error=float(np.max(np.abs(error))),
                    derivative_status=status, derivative_valid=valid,
                    derivative_mask=None if dim is None else np.full((10, dim), valid, dtype=bool),
                    complement_scaled_error=complement,
                    maximum_complement_scaled_error=float(np.max(np.abs(complement))),
                    complement_jacobian_scaled_error=None,
                    jacobian_dimensionless_error=None, actual_direction_Jv_scaled_error=None,
                    unit_direction_Jv_scaled_error=None,
                    direction_status='MISSING' if valid else 'DERIVATIVE_UNAVAILABLE')
        if valid:
            parameter_scales = np.asarray(case['parameter_scales'])
            item['jacobian_dimensionless_error'] = (j-native_j)*parameter_scales[None,:]/scale[:,None]
            item['complement_jacobian_scaled_error'] = (j+opposite_j)*parameter_scales[None,:]/scale[:,None]
            if case.get('direction') is not None:
                direction = np.asarray(case['direction'])
                norm = float(np.linalg.norm(direction/parameter_scales))
                item['actual_direction_Jv_scaled_error'] = (j-native_j)@direction/scale
                if norm > 0:
                    item['unit_direction_Jv_scaled_error'] = (j-native_j)@(direction/norm)/scale
                    item['direction_status'] = 'AVAILABLE'
                else:
                    item['direction_status'] = 'ZERO_DIRECTION_NOT_INFORMATIVE'
                item['direction_parameter_norm'] = norm
                item['direction_source'] = meta['direction_source']
        records.append(item)
    if operand_fingerprint(operand) != meta['operand_content_sha256']:
        raise ValueError("operand changed during a supposedly frozen comparison")
    synthetic = meta['reference_kind'] == 'analytic_fixture'
    return jsonable(dict(schema='tvcg.v21.native-band-sweep.v2',
                         status='COMPARISON_COMPLETE_NOT_PHYSICS_ACCEPTANCE',
                         scope=('synthetic_geometry_actual_v21_code_path' if synthetic else
                                'actual_v21_no_weights_same_operand_not_exact_boolean'),
                         native_solver_acceptance='NOT_EVALUATED', model_weights_used=False,
                         provenance=meta, asset_receipts=case.get('asset_receipts'),
                         manifest_receipt=case.get('manifest_receipt'), fixed_moments=fixed,
                         material_audit=audit, load_and_pack_seconds=case.get('load_and_pack_seconds'),
                         material_audit_seconds=audit_seconds, surface_setup_seconds=setup_seconds,
                         timing_scope='query_seconds_exclude_input_load_material_audit_and_surface_setup',
                         direction_normalization='chart: v_unit = v / norm(v / parameter_scales); not raw-coefficient length normalization',
                         jacobian_error_normalization='(J-J_reference) * parameter_scales[None,:] / (editable_mass * length_scale**degree)[:,None]',
                         records=records))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',required=True)
    source = p.add_mutually_exclusive_group()
    source.add_argument('--adapter', help='Work module:factory returning a validated baseline case')
    source.add_argument('--case', help='frozen baseline-case JSON; existing material/operand/target assets only')
    source.add_argument('--write-synthetic-case', metavar='NEW_DIRECTORY',
                        help='export an explicitly synthetic finite fixture, then run the actual V2.1 path')
    p.add_argument('--cache-directory', help='optional disk-backed material block cache parent for --case')
    p.add_argument('--chunk-size', type=int, default=65536)
    a=p.parse_args()
    if a.cache_directory and not (a.case or a.write_synthetic_case):
        p.error('--cache-directory requires --case or --write-synthetic-case')
    if a.write_synthetic_case:
        case_path = write_synthetic_finite_case(a.write_synthetic_case)
        result = v21_band_sweep(load_baseline_case(case_path, cache_directory=a.cache_directory), chunk_size=a.chunk_size)
    elif a.case:
        result = v21_band_sweep(load_baseline_case(a.case, cache_directory=a.cache_directory), chunk_size=a.chunk_size)
    elif a.adapter:
        result = v21_band_sweep(load_factory(a.adapter)(), chunk_size=a.chunk_size)
    else:
        result = synthetic_suite()
    with open(a.output,'x',encoding='utf8') as f: json.dump(result,f,indent=2,allow_nan=False); f.write('\n')
    return 2 if result.get('status') == 'STOP_FULL_MATERIAL_REFERENCE' else 0


if __name__=='__main__': raise SystemExit(main())
