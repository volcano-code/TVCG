"""Sharded physical replay of the complete registered development R3/R8 population."""
import argparse
from contextlib import contextmanager
from dataclasses import replace
from pathlib import Path
import multiprocessing
import os
import resource
import sys
from time import perf_counter

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
sys.path[:0] = [str(ROOT), str(HERE), str(HERE.parents[1]/"native")]
import checks
from experiments.compact_pino_v21.stratum.shared_operator import SharedStratumOperator
from experiments.compact_pino_v21.stratum.primal import retained_value
from registry import Registry, RawChart, read, save_json, save_npz, sha
from replay import DEFAULT_ALIGNMENT, DEFAULT_CORPUS, DEFAULT_PHYSICS, NATIVE_EXCEPTIONS
from v21_native import FiniteOperand, operand_from_source, physical_operand_vertices
from v21_native_physics import BoundaryWitness, NativePhysicalTargets, evaluate_operand_physics


ASSETS = tuple(checks.PROTOCOL["population"]["selectors"])
CONTROL_KEY = "archived_alternative_cross_candle"
ERRORS = NATIVE_EXCEPTIONS + (FloatingPointError,)


def observe_error(fn):
    try:
        return fn(), None
    except ERRORS as exc:
        return None, dict(exception=type(exc).__name__, diagnostic=str(exc))


def value_condition(operator,q,chart):
    if not hasattr(operator,"extension"):
        operand,error=observe_error(lambda:operator.condition(q,chart,with_motion=False))
        return operand,error,None
    extension=operator.extension
    info={}
    class ObserveNative:
        def source_value(self,*arguments):
            packet=extension.source_value(*arguments)
            for name in ("source_map_version","surface_type","side_selection_method","interior_is_positive",
                         "truncation_radius","triangulation_hash","interface_vertex_count","interface_face_count",
                         "cap_face_count","construction_witness"):
                if name in packet:
                    info[name]=packet[name]
            return packet
    operator.extension=ObserveNative()
    try:
        operand,error=observe_error(lambda:operator.condition(q,chart,with_motion=False))
    finally:
        operator.extension=extension
    return operand,error,info or None


def source_facts(operand):
    return dict(operation=operand.operation, n_vertices=len(operand.vertices), n_faces=len(operand.faces),
                n_cap_faces=int(operand.face_is_cap.sum()), vertices_sha256=checks.digest(operand.vertices, "<f8"),
                faces_sha256=checks.digest(operand.faces, "<i8"), cap_sha256=checks.digest(operand.face_is_cap,"u1"))


def source_match(base, other):
    return dict(operation=base.operation == other.operation,
                vertex_shape=base.vertices.shape == other.vertices.shape,
                faces=bool(np.array_equal(base.faces, other.faces)),
                caps=bool(np.array_equal(base.face_is_cap, other.face_is_cap)))


def input_physical(backend, operand):
    r = backend.request
    return physical_operand_vertices(operand, r.target_axes, r.target_origin, backend.length, backend.normalization_origin)


@contextmanager
def capture_boolean(backend):
    original = backend._boolean_with_valid_source
    capture = {}
    def capture_call(vertices, faces, operation):
        capture.update(input_vertices_sha256=checks.digest(vertices, "<f8"), input_faces_sha256=checks.digest(faces, "<i8"),
                       operation=operation)
        capture["packet"] = original(vertices, faces, operation)
        return capture["packet"]
    backend._boolean_with_valid_source = capture_call
    try:
        yield capture
    finally:
        del backend._boolean_with_valid_source


class HistoricalControl:
    """Alternative archive-only diagnostic. It never participates in the 89-row replay."""
    def __init__(self, reg, geom):
        from make_it_stand.verification.cherchi_inprocess import InProcessCherchiEvaluator
        self.backend=InProcessCherchiEvaluator(geom.request, reg.panel.spin.SpinEvaluatorConfig(
            **dict(reg.panel.profile, algebraic_finite_map=False)),
            normalization=geom.normalization, design_vertices_physical=geom.physical,
            rational_arithmetic_backend="NATIVE_GMP_SURFACE_V1", fixed_boundary_reuse=True,
            integral_profile="CERTIFIED_MPFR256_GMP_FALLBACK_V1")
        self.identity = self.backend.context_identity

    def condition(self, q, chart, *, with_motion=True):
        b = self.backend
        s = b.native.evaluate_spin_it_faster_stratum_source_motion(q,b.target_design,b.triangulation)
        if with_motion:
            return operand_from_source(s, q, chart=chart)
        if not s["defined"] or not s["side_selection_defined"]:
            raise ValueError(s["diagnostic"])
        return FiniteOperand(s["vertices"],s["faces"],np.asarray(s["face_is_cap"]).reshape(-1),
                             "intersection" if s["interior_is_positive"] else "difference")

    def teacher(self, q, chart, *, capture_boundary):
        return evaluate_operand_physics(self.backend, self.condition(q,chart), chart=chart,
                                       capture_boundary=capture_boundary)


def get_retained(backend, operand, target, capture, design_index):
    probe = {k:v for k,v in capture.items() if k != "packet"}
    if "operation" in capture:
        probe["same_actual_physical_operand"] = bool(
            capture["operation"] == operand.operation and capture["input_faces_sha256"] == checks.digest(operand.faces,"<i8")
            and capture["input_vertices_sha256"] == checks.digest(input_physical(backend,operand),"<f8"))
        if not probe["same_actual_physical_operand"]:
            raise ValueError("changed physical input between conditioning and native Boolean")
    record = dict(moments=target.moments, editable_moments=target.editable_moments,
                  moment_valid=target.moment_valid, moment_diagnostic=target.moment_diagnostic,
                  derivative_diagnostic=target.derivative_diagnostic,
                  material_geometry_hash=target.geometry_hash, material_topology_hash=target.topology_hash, actual_boolean_input=probe)
    if not target.moment_valid:
        return record, None
    w = target.boundary_witness
    if w is None or w.vertices is None or w.faces is None:
        raise RuntimeError("numeric material lacks retained boundary witness")
    record.update(retained_n_vertices=len(w.vertices), retained_n_faces=len(w.faces),
                  retained_vertices_sha256=checks.digest(w.vertices,"<f8"),
                  retained_faces_sha256=checks.digest(w.faces,"<i8"))
    if "packet" not in capture:
        raise RuntimeError("Boolean produced a value without an observable packet")
    source_ids = checks.PhysicalVertexIDs(design_index, input_physical(backend,operand))
    ids, err = observe_error(lambda: source_ids.evaluate(capture["packet"],w.faces))
    record["correspondence_error"] = err
    if ids is not None:
        record["retained_signature"] = ids.summary()
    return record, ids


def rejection_facts(backend, q, probe):
    result = {}
    for name, evaluate in (("typed",backend.native.evaluate_spin_it_faster_stratum_source_motion),
                           ("ambient",backend.native.evaluate_spin_it_faster_source_motion)):
        s, error = observe_error(lambda:evaluate(q,backend.target_design,backend.triangulation))
        result[name] = (dict(defined=False, error=error, diagnostic=error["diagnostic"]) if error else
                        dict(defined=bool(s.get("defined")), diagnostic=s.get("diagnostic"),
                             side_defined=bool(s.get("side_selection_defined"))))
    return result


def endpoint(backend, operator, q, chart, ref, design_index, method):
    op, err, native = value_condition(operator,q,chart)
    result = dict(query=q, source_error=err, native_value=native)
    if op is None:
        return result, None, None, None
    result.update(source=source_facts(op), source_match=source_match(ref,op))
    if method == "reference":
        with capture_boolean(backend) as capture:
            target = evaluate_operand_physics(backend, op, capture_boundary=True)
    else:
        with capture_boolean(backend) as capture:
            measured, err = observe_error(lambda:retained_value(backend,op))
        if err is not None:
            target=NativePhysicalTargets(moment_diagnostic=err["diagnostic"])
        else:
            target=NativePhysicalTargets(moments=measured.moments, editable_moments=measured.moments-backend.fixed,
                                         boundary_witness=BoundaryWitness(measured.vertices,measured.faces))
            result.update(value_path=measured.path, arithmetic_receipt=measured.arithmetic)
            if "packet" not in capture or capture["packet"] is not measured.packet:
                raise ValueError("fast value did not return its actual physical Boolean packet")
    record, ids = get_retained(backend, op, target, capture, design_index)
    result["material"] = record
    return result, op, ids, target.boundary_witness


def center(out, geom, probe, backend, legacy, operator, design_index, key, method):
    q, chart = probe.query,probe.chart
    dim = chart.basis.shape[1]
    row = dict(schema="compact-pino.algebraic-stratum.registered-row.v1", row_id=probe.id,
               geometry_id=geom.id, split=geom.split, source_record=probe.origin,
               query=q, raw_basis=chart.basis, chart=chart.surface, dimension=dim,
               protocol_sha256=sha(checks.PROTOCOL_FILE), context=operator.identity,
               registered_physics_context=backend.context_identity, legacy_context=legacy.context_identity,
               base_algebraic_finite_map=bool(backend.triangulation.algebraic_finite_map),
               experiment_key=key, finite_difference_integrator=method,
               ambient_event_valid=-1, restricted_event_valid=-1, full_j_valid=False,
               chart_geometry=checks.chart_facts(q,chart), legacy_source=rejection_facts(legacy,q,probe),
               boolean_backend=backend.backend_identity, arithmetic=backend.arithmetic_identity,
               integral_profile=backend.integral_profile, density=backend.request.material_density,
               fixed_moments=backend.fixed, geometry_files=geom.hashes,
               geometry_frame=dict(normalization=geom.normalization,target_origin=geom.request.target_origin,
                                   target_axes=geom.request.target_axes), source=None, material=None)
    op, motion_error = observe_error(lambda:operator.condition(q,chart,with_motion=True))
    value, value_error, native = value_condition(operator,q,chart)
    row.update(source_motion_error=motion_error, primal_error=value_error, native_value=native)
    if op is not None:
        path = out/"operands"/(probe.id+".npz")
        path.parent.mkdir(parents=True,exist_ok=True)
        op.save(path)
        row.update(source=source_facts(op), operand_file="operands/"+path.name, operand_sha256=sha(path),
                   raw_source_velocity_sha256=checks.digest(op.vertex_velocity,"<f8"))
    if value is not None:
        row["primal"] = source_facts(value)
    if value is not None and op is not None:
        row["same_primal_motion_source"] = bool(all(source_match(op,value).values()) and
                                                np.array_equal(op.vertices,value.vertices))
    with capture_boolean(backend) as cap:
        target = operator.teacher(q,chart,capture_boundary=True)
    if target.evaluation_map_identity != operator.identity:
        raise ValueError("typed material result used a different versioned identity")
    if target.full_j_valid:
        raise ValueError("typed chart teacher exposed an ambient Jacobian")
    physical_op = op if op is not None else value
    if physical_op is None:
        if target.moment_valid:
            raise ValueError("teacher accepted a source unavailable from its own value and motion routes")
        row["material"] = dict(moment_valid=False,moment_diagnostic=target.moment_diagnostic,
                               derivative_diagnostic=target.derivative_diagnostic)
        return row,None,None,None
    material, ids = get_retained(backend,physical_op,target,cap,design_index)
    material["restricted_valid"]=target.restricted_valid
    row["material"]=material
    if target.restricted_valid and not np.array_equal(target.chart.basis,chart.basis):
        raise ValueError("typed teacher changed the raw chart column order")
    if target.moment_valid:
        w=target.boundary_witness
        mt,jt=checks.tetra_pair(w.vertices,w.faces,w.vertex_velocity,backend.request.material_density,backend.fixed)
        material.update(M_independent_tetra=mt, moment_tetra=checks.tetra_error(mt,target.moments))
        if target.restricted_valid:
            if w.vertex_velocity is None or target.restricted_jacobian.shape != (10,dim):
                raise ValueError("derivative lacks a dimension-matched retained velocity")
            material.update(raw_restricted_J=target.restricted_jacobian, J_independent_tetra=jt,
                            jacobian_tetra=checks.tetra_error(jt,target.restricted_jacobian))
        witness_path=out/"boundaries"/(probe.id+".npz")
        witness_path.parent.mkdir(parents=True,exist_ok=True)
        arrays=dict(vertices=w.vertices,faces=w.faces)
        if w.vertex_velocity is not None: arrays["vertex_velocity"]=w.vertex_velocity
        if ids is not None: arrays["construction_keys"]=ids.keys
        save_npz(witness_path,**arrays)
        material.update(retained_boundary_file="boundaries/"+witness_path.name,
                        retained_boundary_sha256=sha(witness_path))
    return row,physical_op,target,ids


def step_column(probe, backend, operator, base, target, ids, design_index, h, k, method):
    q,chart=probe.query,probe.chart
    direction=chart.basis[:,k]
    sides=[]
    data=[]
    for sign in (-1,1):
        q1=checks.retracted(q,direction,sign*h,chart.surface)
        sig=checks.rank_signature(q1,chart.surface)
        if sig != checks.rank_signature(q,chart.surface):
            raise ValueError("probe crossed the registered numerical rank/signature")
        tick=perf_counter()
        s, cut, nodes, witness=endpoint(backend,operator,q1,chart,base,design_index,method)
        s.update(sign=sign, quadratic_negative_positive=sig, elapsed_seconds=perf_counter()-tick)
        sides.append(s); data.append((cut,nodes,witness))
    out=dict(h=h, raw_col=k, endpoints=sides,
             path_tangent_max_abs=float(np.max(abs((sides[1]["query"]-sides[0]["query"])/(2*h)-direction))))
    if all(cut is not None for cut,_,_ in data):
        stable=all(all(s["source_match"].values()) for s in sides)
        out["source_combinatorics_stable"]=stable
        if stable and base.vertex_velocity is not None:
            dv=(data[1][0].vertices-data[0][0].vertices)/(2*h)
            out["source_velocity"]=checks.error_metric(dv,base.vertex_velocity[...,k],"source")
    if all(s.get("material",{}).get("moment_valid") for s in sides):
        fd=(sides[1]["material"]["moments"]-sides[0]["material"]["moments"])/(2*h)
        out["material_FD"]=fd
        if target is not None and target.restricted_valid:
            out["material_jacobian"]=checks.error_metric(fd,target.restricted_jacobian[:,k],"material")
    if ids is not None and all(d[1] is not None for d in data):
        velocity=None
        if target.boundary_witness.vertex_velocity is not None:
            velocity=target.boundary_witness.vertex_velocity[...,k]
        out["retained"]=checks.compare_retained(ids,data[0][1],data[1][1],
                    data[0][2].vertices,data[1][2].vertices,velocity,h)
    out["passed_sampled_column"]=bool(out.get("source_combinatorics_stable") and out.get("source_velocity",{}).get("passed") and
        out.get("material_jacobian",{}).get("passed") and out.get("retained",{}).get("complete") and
        out.get("retained",{}).get("velocity",{}).get("passed"))
    return out


def fingerprint_run(operator, probe, method):
    return dict(protocol_sha256=sha(checks.PROTOCOL_FILE), context=operator.identity,
                row_id=probe.id, query_sha256=checks.digest(probe.query,"<f8"),
                raw_basis_sha256=checks.digest(probe.chart.basis,"<f8"), integrator=method,
                replay_code_sha256=sha(Path(__file__)), checks_code_sha256=sha(checks.__file__),
                primal_code_sha256=sha(HERE.parent/"primal.py") if method=="fast" else None,
                primal_transform_sha256=sha(HERE.parent/"primal_transform.cpp") if method=="fast" else None)


def revive(out, row, query, chart):
    m=row["material"]
    target=NativePhysicalTargets(
        moments=np.asarray(m["moments"]) if m.get("moment_valid") else None,
        restricted_jacobian=np.asarray(m["raw_restricted_J"]) if m.get("restricted_valid") else None, chart=chart)
    boundary_file=m.get("retained_boundary_file")
    ids=None
    if boundary_file:
        path=out/boundary_file
        if sha(path)!=m["retained_boundary_sha256"]:
            raise ValueError("center boundary checkpoint changed")
        with np.load(path,allow_pickle=False) as data:
            w=BoundaryWitness(data["vertices"],data["faces"],data["vertex_velocity"] if "vertex_velocity" in data else None)
            target.boundary_witness=w
            if "construction_keys" in data:
                ids=checks.retain_keys(data["construction_keys"],w.faces)
    op=None
    if row.get("operand_file"):
        path=out/row["operand_file"]
        if sha(path)!=row["operand_sha256"]:
            raise ValueError("center operand checkpoint changed")
        op=FiniteOperand.load(path)
    return op,target,ids


def center_checkpoint(out, geom, probe, backend, legacy, operator, design_index, key, method):
    path=out/"centers"/(probe.id+".json")
    expected=fingerprint_run(operator,probe,method)
    if path.is_file():
        row=read(path)
        if row["run_fingerprint"]!=expected:
            raise ValueError("existing center has different identity, chart, protocol or instrumentation")
        op,target,ids=revive(out,row,probe.query,probe.chart)
        if op is None and row.get("primal"):
            op=operator.condition(probe.query,probe.chart,with_motion=False)
        return row,op,target,ids
    start=perf_counter()
    row,op,target,ids=center(out,geom,probe,backend,legacy,operator,design_index,key,method)
    row.update(center_seconds=perf_counter()-start,run_fingerprint=expected)
    save_json(path,row)
    print("center",probe.id, row["material"].get("moment_valid"),row["material"].get("restricted_valid"),row["center_seconds"],flush=True)
    return row,op,target,ids


def execute_probe(out, geom, probe, backend, legacy, operator, design_index, key, method, mode):
    rowpath=out/"rows"/(probe.id+".json")
    expected=fingerprint_run(operator,probe,method)
    if rowpath.exists():
        row=read(rowpath)
        if row.get("run_fingerprint") != expected:
            raise ValueError("existing completed row has a different run fingerprint")
        if row.get("completed"):
            print("resumed",probe.id,"complete",flush=True)
            return
    start=perf_counter()
    row,base,target,ids=center_checkpoint(out,geom,probe,backend,legacy,operator,design_index,key,method)
    if mode=="centers":
        return
    columns=[]
    if base is not None:
        for step_i,h in enumerate(checks.FD["raw_steps"]):
            for k in range(row["dimension"]):
                path=out/"parts"/probe.id/(f"s{step_i}_c{k:02d}.json")
                if path.is_file():
                    part=read(path)
                    if part["run_fingerprint"] != expected:
                        raise ValueError("existing column has a different run fingerprint")
                else:
                    part=step_column(probe,backend,operator,base,target,ids,design_index,h,k,method)
                    part["run_fingerprint"]=expected
                    save_json(path,part)
                columns.append(part)
                print("column",probe.id,step_i,k,part["passed_sampled_column"],flush=True)
    row["steps"]=[]
    for h in checks.FD["raw_steps"]:
        c=[v for v in columns if v["h"]==h]
        summary=dict(h=h, columns=c, n_planned=row["dimension"], n_attempted=len(c))
        if len(c)==row["dimension"] and all("material_FD" in x for x in c):
            fd=np.stack([x["material_FD"] for x in c],axis=1)
            summary["raw_FD_matrix"]=fd
            if target is not None and target.restricted_valid:
                summary["matrix_error"]=checks.error_metric(fd,target.restricted_jacobian,"material")
        summary["all_columns_passed"]=bool(len(c)==row["dimension"] and all(x["passed_sampled_column"] for x in c))
        row["steps"].append(summary)
    material=row["material"]
    row["sampled_row_passed"]=bool(row.get("same_primal_motion_source") and material.get("moment_tetra",{}).get("passed")
         and material.get("jacobian_tetra",{}).get("passed") and all(x["all_columns_passed"] for x in row["steps"]))
    row.update(completed=True, elapsed_seconds=perf_counter()-start,
               peak_RSS_MiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024)
    save_json(rowpath,row)
    print("finished",probe.id,row["sampled_row_passed"],round(row["elapsed_seconds"],3),flush=True)


def _apply(job):
    probe,mode=job
    out, geom, backend, legacy, operator, design_index, key, method=WORK
    return execute_probe(out,geom,probe,backend,legacy,operator,design_index,key,method,mode)


def run(args):
    reg=Registry(DEFAULT_ALIGNMENT, DEFAULT_CORPUS, DEFAULT_PHYSICS)
    legacy_key=reg.panel.cross_ids[0] if args.asset==CONTROL_KEY else args.asset
    geom=reg.example(legacy_key)
    if geom.split != "development":
        raise ValueError("only the registered development geometry can enter the replay")
    legacy=reg.panel.direct(geom.request,geom.normalization,np.asarray(geom.physical).tolist())
    probes=[p for p in geom.probes if p.chart is not None]
    if args.asset==CONTROL_KEY:
        operator=HistoricalControl(reg,geom);backend=operator.backend
        probes=[replace(p,query=2.75*p.query) for p in probes]
    else:
        operator=SharedStratumOperator.from_evaluator(legacy);backend=operator.backend
        counts=[sum(p.chart.surface==s for p in probes) for s in ("PLANE","PARABOLOID")]
        if counts != checks.PROTOCOL["population"]["selectors"][geom.id] or not backend.triangulation.algebraic_finite_map:
            raise ValueError("registered geometry/stratum or algebraic policy differs")
    if args.only and not set(args.only)<=set(p.id for p in probes):
        raise ValueError("unknown registered probe")
    selected=[p for i,p in enumerate(probes) if i%args.shards==args.shard and (not args.only or p.id in args.only)]
    design_index=checks.PhysicalVertexIDs.index_design(backend.design)
    out=Path(args.out).resolve()
    global WORK
    WORK=(out,geom,backend,legacy,operator,design_index,args.asset,args.integrator)
    selected.sort(key=lambda p:(-p.chart.basis.shape[1],p.id))
    runtime=dict(asset=geom.id,geometry_hashes=geom.hashes,registered_context=legacy.context_identity,
                 physical_context=backend.context_identity,versioned_context=operator.identity,
                 source_identity=operator.extension.backend_identity if hasattr(operator,"extension") else None,
                 native_libs={b.__name__:dict(path=b.__file__,sha256=sha(b.__file__)) for b in
                             (backend.native,backend.bridge,backend.arithmetic)},
                 requested_mode=args.mode,physics_profile=reg.panel.profile)
    save_json(out/"geometries"/(geom.id+".json"),runtime)
    modes=["centers","complete"] if args.mode=="centers-first" else [args.mode]
    if args.workers==1:
        for stage in modes:
            for probe in selected:
                _apply((probe,stage))
    else:
        with multiprocessing.get_context("fork").Pool(args.workers) as pool:
            for stage in modes:
                for _ in pool.imap_unordered(_apply,[(p,stage) for p in selected],chunksize=1):
                    pass


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--asset",required=True,choices=(*ASSETS,CONTROL_KEY))
    parser.add_argument("--shard",type=int,default=0)
    parser.add_argument("--shards",type=int,default=1)
    parser.add_argument("--only",nargs="+")
    parser.add_argument("--out",required=True)
    parser.add_argument("--mode",choices=["centers","complete","centers-first"],default="complete")
    parser.add_argument("--workers",type=int,default=1)
    parser.add_argument("--integrator",choices=["reference","fast"],default="reference")
    args=parser.parse_args()
    if not 0<=args.shard<args.shards:
        parser.error("shard must be in [0,shards)")
    run(args)
