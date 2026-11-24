"""Package or verify absolute raw coefficients, original charts, physical measurements and missing data."""
import argparse
from collections import Counter
import gzip
import hashlib
import json
import math
from pathlib import Path
import re
from tempfile import NamedTemporaryFile

import numpy as np

HERE = Path(__file__).resolve().parent
NATIVE_EVIDENCE = HERE.parents[1] / "native" / "evidence"
NAME = "registered_physical_replay.json.gz"
PATTERN = re.compile(r"^(maskB|combB|R01|R02)_R(3|8)_\d+$")


def sha(path):
    with path.open("rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()


def historic():
    archive = json.loads((NATIVE_EVIDENCE / "native_replay.json").read_bytes())
    subset = {r["row_id"]:r for r in archive["rows"] if PATTERN.fullmatch(r["row_id"])}
    with np.load(NATIVE_EVIDENCE / "native_targets.npz", allow_pickle=False) as bank:
        raw = {str(k): q.copy() for k,q in zip(bank["row_id"], bank["raw_query"], strict=True) if str(k) in subset}
    if len(subset) != 89 or raw.keys() != subset.keys():
        raise ValueError("historical registered roster is incomplete")
    return subset,raw


def close_tetra(a,b,protocol):
    return np.allclose(a,b,rtol=protocol["base"]["tetra_elementwise_rtol"],atol=protocol["base"]["tetra_elementwise_atol"])


def quadratic_spectrum(q):
    return np.linalg.eigvalsh([[q[4],q[9],q[8]],[q[9],q[5],q[7]],[q[8],q[7],q[6]]])


def metrics(a,b,scope,protocol):
    x,y=np.asarray(a),np.asarray(b)
    scale=max(np.linalg.norm(x),np.linalg.norm(y))
    err=np.linalg.norm(x-y)
    fd=protocol["central_difference"]
    return bool(err<=fd[scope+"_fd_atol_norm"]+fd[scope+"_fd_rtol_norm"]*scale)


def closed_euler(material):
    if not material.get("moment_valid"):
        return None
    nv,nf=material["retained_n_vertices"],material["retained_n_faces"]
    if nf%2:
        raise ValueError("odd triangle count on normalized closed retained boundary")
    return nv-nf//2


def include_topology(row,center):
    material=row["material"]
    if material["moment_valid"]:
        if not center or center["retained_boundary_sha256"]!=material["retained_boundary_sha256"]:
            raise ValueError("fresh center incidence measurement has a different retained mesh")
        material["center_surface_components"]=center
        if center["euler_characteristic"]!=closed_euler(material):
            raise ValueError("actual edge count differs from closed-mesh invariant")
    for step in row["steps"]:
        for column in step["columns"]:
            values=[closed_euler(side.get("material",{})) for side in column["endpoints"]]
            column["closed_euler_center_minus_plus"]=[closed_euler(material),*values]


def check_sampled_aggregates(row):
    dim=row["dimension"]
    verified=[]
    for step in row["steps"]:
        columns=step["columns"]
        if step["n_planned"]!=dim or step["n_attempted"]!=len(columns):
            raise ValueError(f"step aggregate direction count disagrees at {row['row_id']}")
        result=len(columns)==dim and all(c["passed_sampled_column"] is True for c in columns)
        if step["all_columns_passed"] is not result:
            raise ValueError(f"step aggregate disagrees with its measured columns at {row['row_id']}")
        verified.append(result)
    material=row["material"]
    result=bool(row.get("same_primal_motion_source") and material.get("moment_tetra",{}).get("passed")
                and material.get("jacobian_tetra",{}).get("passed") and verified and all(verified))
    if row["sampled_row_passed"] is not result:
        raise ValueError(f"row aggregate disagrees with its source, tetrahedra or measured steps at {row['row_id']}")


def check_discrete_receipts(row, column):
    key = row["row_id"]
    source = row.get("source") or row.get("primal")
    endpoints = column["endpoints"]
    matches = []
    for endpoint in endpoints:
        other = endpoint.get("source")
        if source is None or other is None:
            continue
        match = dict(operation=other["operation"] == source["operation"],
                     vertex_shape=other["n_vertices"] == source["n_vertices"],
                     faces=other["n_faces"] == source["n_faces"] and other["faces_sha256"] == source["faces_sha256"],
                     caps=other["n_cap_faces"] == source["n_cap_faces"] and other["cap_sha256"] == source["cap_sha256"])
        if endpoint["source_match"] != match:
            raise ValueError(f"source side, shape, face or cap receipt disagrees at {key}")
        matches.append(all(match.values()))
    if len(matches) == 2 and column.get("source_combinatorics_stable") is not all(matches):
        raise ValueError(f"source combinatorics disagree with the two observed operands at {key}")

    signatures = [material.get("retained_signature") for material in
                  (row["material"], *(endpoint.get("material", {}) for endpoint in endpoints))]
    retained = column.get("retained")
    if not all(signatures):
        if retained is not None:
            raise ValueError(f"retained comparison lacks a measured boundary signature at {key}")
        return
    if not all(s["unique_constructions"] for s in signatures):
        if retained is None or retained.get("complete") is not False or retained.get("n_matched") != 0:
            raise ValueError(f"ambiguous retained constructions reported as corresponding at {key}")
        return
    base, *sides = signatures
    same_ids = all(s["construction_sha256"] == base["construction_sha256"] for s in sides)
    same_faces = same_ids and all(s["unoriented_incidence_sha256"] == base["unoriented_incidence_sha256"] for s in sides)
    same_oriented = same_faces and all(s["oriented_incidence_sha256"] == base["oriented_incidence_sha256"] for s in sides)
    expected = {"same_construction_set": same_ids, "same_unoriented_incidence": same_faces,
                "same_oriented_incidence": same_oriented, "complete": same_oriented}
    if retained is None or any(retained.get(field) is not value for field, value in expected.items()):
        raise ValueError(f"retained correspondence disagrees with the signed boundary signatures at {key}")
    if (retained["n_center"] != base["vertex_count"] or not 0 <= retained["n_matched"] <= min(s["vertex_count"] for s in signatures)
            or (same_ids and retained["n_matched"] != base["vertex_count"])):
        raise ValueError(f"retained matching counts disagree with the signed boundary signatures at {key}")


def check_row(row, old, q, protocol, source_check=False):
    key=row["row_id"]
    dim=int(PATTERN.fullmatch(key).group(2))
    if (row["split"]!="development" or row["dimension"]!=dim or row["chart"]!=old["requested_chart"]
            or row["source_record"]!=old["source_record"] or not np.array_equal(q,row["query"])
            or row["ambient_event_valid"]!=-1 or row["restricted_event_valid"]!=-1
            or not row["completed"] or row["full_j_valid"] or not row["base_algebraic_finite_map"]):
        raise ValueError(f"registration, scope or completion mismatch at {key}")
    b=np.asarray(row["raw_basis"])
    fingerprint=row["run_fingerprint"]
    hash_array=lambda x:hashlib.sha256(np.ascontiguousarray(x,dtype="<f8").tobytes()).hexdigest()
    if (fingerprint["context"]!=row["context"] or fingerprint["query_sha256"]!=hash_array(q)
            or fingerprint["raw_basis_sha256"]!=hash_array(b) or fingerprint["row_id"]!=key
            or fingerprint["protocol_sha256"]!=row["protocol_sha256"]):
        raise ValueError(f"center input fingerprint changed at {key}")
    if b.shape!=(10,dim) or np.linalg.matrix_rank(b)!=dim:
        raise ValueError(f"invalid ordered raw tangent at {key}")
    if source_check:
        src=row["source_record"]["source_chart"]
        path=Path(src["file"])
        i=src["index"]
        if sha(path)!=src["sha256"]:
            raise ValueError(f"chart archive digest changed at {key}")
        with np.load(path,allow_pickle=False) as z:
            if (str(z["query_id"][i])!=src["query_id"] or not np.array_equal(z["queries"][i],q)
                    or not np.array_equal(z["restricted_basis"][i,:,:dim],b)):
                raise ValueError(f"archived raw basis or query disagrees at {key}")
    if row["legacy_source"]["typed"]["defined"] or row["legacy_source"]["ambient"]["defined"]:
        raise ValueError(f"legacy rejection behavior changed at {key}")
    if row["legacy_source"]["typed"]["diagnostic"]!=old["source_error"]["diagnostic"]:
        raise ValueError(f"frozen typed source rejection disagrees at {key}")
    if row.get("source") is not None and row.get("primal") is not None:
        if row.get("same_primal_motion_source") is not (row["source"] == row["primal"]):
            raise ValueError(f"primal and motion center source receipts disagree at {key}")
    m=row["material"]
    if m["moment_valid"]:
        if (np.asarray(m["moments"]).shape!=(10,) or not np.isfinite(m["moments"]).all()
                or m["moment_tetra"]["passed"]!=close_tetra(m["M_independent_tetra"],m["moments"],protocol)):
            raise ValueError(f"invalid retained integrated moments or independent check at {key}")
        if not m["actual_boolean_input"]["same_actual_physical_operand"]:
            raise ValueError(f"physical Boolean selected a different source operand at {key}")
    elif not m.get("moment_diagnostic") or m.get("moments") is not None:
        raise ValueError(f"missing refusal reason or fabricated material value at {key}")
    j=m.get("raw_restricted_J")
    if m.get("restricted_valid"):
        if (np.asarray(j).shape!=(10,dim) or not np.isfinite(j).all() or
                m["jacobian_tetra"]["passed"]!=close_tetra(m["J_independent_tetra"],j,protocol)):
            raise ValueError(f"invalid retained differential or independent check at {key}")
    elif j is not None:
        raise ValueError(f"unavailable chart derivative has numeric entries at {key}")
    steps=protocol["central_difference"]["raw_steps"]
    if [x["h"] for x in row["steps"]]!=steps:
        raise ValueError(f"missing registered step at {key}")
    for s in row["steps"]:
        columns=s["columns"]
        n=dim if row.get("source") or row.get("primal") else 0
        if [c["raw_col"] for c in columns]!=list(range(n)):
            raise ValueError(f"missing raw columns at {key}")
        for k,c in enumerate(columns):
            endpoints=c["endpoints"]
            check_discrete_receipts(row,c)
            if c["closed_euler_center_minus_plus"]!=[closed_euler(m),*[closed_euler(e.get("material",{})) for e in endpoints]]:
                raise ValueError(f"closed-mesh Euler count mismatch at {key}")
            if c["run_fingerprint"]!=fingerprint or c["h"]!=s["h"]:
                raise ValueError(f"column checkpoint changed center, implementation, or step at {key}")
            qminus,qplus=(np.asarray(e["query"]) for e in endpoints)
            if (qminus.shape!=(10,) or qplus.shape!=(10,) or not np.isfinite([qminus,qplus]).all()
                    or qminus.shape!=q.shape or
                    abs(float(np.max(abs((qplus-qminus)/(2*s["h"])-b[:,k])))-c["path_tangent_max_abs"])>1e-12):
                raise ValueError(f"published raw endpoint paths differ from the column tangent at {key}")
            if [x["sign"] for x in endpoints]!=[-1,1]:
                raise ValueError(f"missing directional sign at {key}")
            center_inertia = row["chart_geometry"]["quadratic_negative_positive"]
            for endpoint in endpoints:
                eig = quadratic_spectrum(endpoint["query"])
                tol = protocol["central_difference"]["numerical_rank_relative_tolerance"] * np.linalg.norm(endpoint["query"])
                inertia = [int(np.sum(eig < -tol)), int(np.sum(eig > tol))]
                if inertia != center_inertia or inertia != endpoint["quadratic_negative_positive"]:
                    raise ValueError(f"signed endpoint changed the registered numerical inertia at {key}")
            if all(x.get("material",{}).get("moment_valid") for x in endpoints):
                delta=(np.asarray(endpoints[1]["material"]["moments"])-endpoints[0]["material"]["moments"])/(2*s["h"])
                if not np.array_equal(delta,c["material_FD"]):
                    raise ValueError(f"published raw FD differs from new cuts at {key}")
                if m.get("restricted_valid") and c["material_jacobian"]["passed"]!=metrics(delta,np.asarray(j)[:,k],"material",protocol):
                    raise ValueError(f"misreported retained derivative check at {key}")
            elif "material_FD" in c:
                raise ValueError(f"fabricated material difference at {key}")
            for x in endpoints:
                if x.get("material",{}).get("moment_valid") and not x["material"]["actual_boolean_input"]["same_actual_physical_operand"]:
                    raise ValueError(f"perturbed actual source mismatch at {key}")
            want=bool(c.get("source_combinatorics_stable") and c.get("source_velocity",{}).get("passed")
                and c.get("material_jacobian",{}).get("passed") and c.get("retained",{}).get("complete")
                and c.get("retained",{}).get("velocity",{}).get("passed"))
            if bool(c["passed_sampled_column"])!=want:
                raise ValueError(f"misreported complete column at {key}")
        if "raw_FD_matrix" in s:
            if not np.array_equal(s["raw_FD_matrix"],np.asarray([c["material_FD"] for c in columns]).T):
                raise ValueError(f"matrix column order changed at {key}")

    check_sampled_aggregates(row)


def tally(rows):
    scope=Counter()
    failures=Counter()
    counter=Counter()
    maxima={}
    observed_errors={}
    source_shapes=Counter()
    depth_doublings=Counter()
    matched_fractions=[]
    def peak(k,v):
        maxima[k]=max(float(v),maxima.get(k,0.))
    for row in rows:
        scope["registered"]+=1
        m=row["material"]
        src=row.get("source")
        eig = quadratic_spectrum(row["query"])
        nominal_zero = eig[np.argmin(np.abs(eig))]
        peak("center_nearest_quadratic_eigenvalue_absolute", abs(nominal_zero))
        peak("center_chart_tangent_residual", max(row["chart_geometry"]["tangent_residual"]))
        if src is not None:
            source_shapes[(src["operation"],src["n_faces"],src["n_cap_faces"])]+=1
        witness=(row.get("native_value") or {}).get("construction_witness",{})
        if "cap_depth_doublings" in witness:
            depth_doublings[str(witness["cap_depth_doublings"])]+=1
        scope["source_motion"]+=src is not None
        scope["source_value"]+=row.get("primal") is not None
        scope["source_primal_motion_bitwise"]+=bool(row.get("same_primal_motion_source"))
        scope["M"]+=bool(m.get("moment_valid"))
        scope["J"]+=bool(m.get("restricted_valid"))
        scope["tetra_M"]+=bool(m.get("moment_tetra",{}).get("passed"))
        scope["tetra_J"]+=bool(m.get("jacobian_tetra",{}).get("passed"))
        scope["all_scoped_sampled_checks"]+=bool(row.get("sampled_row_passed"))
        if m.get("moment_tetra"): peak("tetra_M_absolute",m["moment_tetra"]["max_abs"])
        if m.get("jacobian_tetra"): peak("tetra_J_absolute",m["jacobian_tetra"]["max_abs"])
        if not m.get("moment_valid"):
            failures["center M — "+m["moment_diagnostic"]]+=1
        elif not m.get("restricted_valid"):
            failures["center J — "+str(m.get("derivative_diagnostic"))]+=1
        cols=[c for step in row["steps"] for c in step["columns"]]
        def measured_euler(c):
            return [closed_euler(m),*[closed_euler(e.get("material",{})) for e in c["endpoints"]]]
        def equal_euler(c):
            values=measured_euler(c)
            return all(v is not None for v in values) and len(set(values))==1
        for prop,fn in (
            ("same_source_side_caps_faces_all_probes",lambda c:c.get("source_combinatorics_stable")),
            ("all_source_velocity_FD",lambda c:c.get("source_velocity",{}).get("passed")),
            ("all_material_columns_FD",lambda c:c.get("material_jacobian",{}).get("passed")),
            ("all_matched_retained_velocity_FD",lambda c:c.get("retained",{}).get("velocity",{}).get("passed")),
            ("all_complete_retained_ids",lambda c:c.get("retained",{}).get("same_construction_set")),
            ("all_retained_topology_and_ids",lambda c:c.get("retained",{}).get("complete")),
            ("all_closed_euler_unchanged",equal_euler),
        ):
            scope[prop]+=bool(len(cols)==2*row["dimension"] and all(fn(c) for c in cols))
        for step in row["steps"]:
            if "matrix_error" in step:
                name=f"h={step['h']}_retained_J_Frobenius_relative"
                peak(name,step["matrix_error"]["relative_l2"])
                observed_errors.setdefault(name,[]).append(step["matrix_error"]["relative_l2"])
        for c in cols:
            counter["attempted_column_steps"]+=1
            peak("retracted_raw_tangent_max_component_error", c["path_tangent_max_abs"])
            counter["source_side_caps_faces_equal"]+=bool(c.get("source_combinatorics_stable"))
            counter["numeric_material_FD"]+="material_FD" in c
            counter["material_J_FD_passed"]+=bool(c.get("material_jacobian",{}).get("passed"))
            counter["source_velocity_FD_passed"]+=bool(c.get("source_velocity",{}).get("passed"))
            retained=c.get("retained",{})
            if retained.get("n_center"):
                matched_fractions.append(retained["n_matched"]/retained["n_center"])
            counter["retained_constructions_equal"]+=bool(retained.get("same_construction_set"))
            counter["retained_unoriented_incidence_equal"]+=bool(retained.get("same_unoriented_incidence"))
            counter["retained_oriented_incidence_equal"]+=bool(retained.get("same_oriented_incidence"))
            counter["retained_triangulation_equal"]+=bool(c.get("retained",{}).get("complete"))
            counter["matched_retained_velocity_FD_passed"]+=bool(c.get("retained",{}).get("velocity",{}).get("passed"))
            euler=measured_euler(c)
            if euler and all(v is not None for v in euler):
                counter["closed_Euler_compared"]+=1
                counter["closed_Euler_equal"]+=len(set(euler))==1
            counter["all_checks_passed"]+=c["passed_sampled_column"]
            for name,d in (("material",c.get("material_jacobian")),("source",c.get("source_velocity")),
                           ("matched_retained_velocity",c.get("retained",{}).get("velocity"))):
                if d:
                    peak(name+"_FD_max_component_error",d["max_abs"]);peak(name+"_FD_relative",d["relative_l2"])
            for e in c["endpoints"]:
                counter["source_endpoint_calls"]+=1
                endpoint_eig = quadratic_spectrum(e["query"])
                peak("endpoint_nearest_eigenvalue_drift_absolute", abs(endpoint_eig[np.argmin(np.abs(endpoint_eig))] - nominal_zero))
                counter["material_endpoint_values"]+=bool(e.get("material",{}).get("moment_valid"))
                if e.get("source_error"): failures["endpoint source — "+e["source_error"]["diagnostic"]]+=1
                if e.get("material",{}).get("moment_diagnostic"): failures["endpoint M — "+e["material"]["moment_diagnostic"]]+=1
    distributions={k:dict(n=len(v),median=float(np.median(v)),p95=float(np.quantile(v,.95)),max=max(v))
                   for k,v in observed_errors.items()}
    if matched_fractions:
        distributions["fraction_of_center_constructions_matched_in_both_signs"]=dict(n=len(matched_fractions),
              minimum=min(matched_fractions),median=float(np.median(matched_fractions)),maximum=max(matched_fractions))
    source_populations=[dict(operation=op,faces=nf,cap_faces=nc,rows=count)
                        for (op,nf,nc),count in sorted(source_shapes.items())]
    return dict(geometry_rows=dict(scope),directional_counts=dict(counter),maxima=maxima,
                center_source_configurations=source_populations,cap_depth_doublings=dict(depth_doublings),
                distributions=distributions,rejection_reasons=dict(failures))


def exceptions(rows):
    result={"numerical_column_failures":[],"closed_euler_changes":[]}
    for row in rows:
        for step in row["steps"]:
            for c in step["columns"]:
                idx=dict(row_id=row["row_id"],h=step["h"],raw_column=c["raw_col"])
                measures=dict(material=c.get("material_jacobian"),source=c.get("source_velocity"),
                              matched_retained=c.get("retained",{}).get("velocity"))
                missing_or_failed={name:d for name,d in measures.items() if d is None or not d["passed"]}
                if missing_or_failed:
                    result["numerical_column_failures"].append(dict(idx,checks=missing_or_failed))
                euler=c["closed_euler_center_minus_plus"]
                if all(v is not None for v in euler) and len(set(euler))!=1:
                    result["closed_euler_changes"].append(dict(idx,center_minus_plus=euler))
    return result


def group_rows(rows, key):
    return [r for r in rows if key=="all" or "R"+str(r["dimension"])==key or r["geometry_id"]==key]


def groups(rows):
    keys=("all","R3","R8","maskB","combB","R01","R02")
    return {k:tally(group_rows(rows,k)) for k in keys}


def verify_summary_groups(recorded, recomputed, rows):
    """Permit binary64 LAPACK rounding only in two postprocessed near-zero eigenvalue maxima."""
    if recorded.keys() != recomputed.keys():
        raise ValueError("physical summary groups disagree")

    def matrix_norm(q):
        # Frobenius norm of the raw symmetric coefficient block, with both copies of every off-diagonal.
        return math.hypot(q[4], q[5], q[6], q[7], q[7], q[8], q[8], q[9], q[9])

    for key,observed in recomputed.items():
        original=recorded[key]
        center_scale=drift_scale=0.
        for row in group_rows(rows,key):
            at_center=matrix_norm(row["query"])
            center_scale=max(center_scale,at_center)
            for step in row["steps"]:
                for col in step["columns"]:
                    for endpoint in col["endpoints"]:
                        drift_scale=max(drift_scale,at_center+matrix_norm(endpoint["query"]))
        maxima=dict(observed["maxima"])
        if maxima.keys()!=original["maxima"].keys():
            raise ValueError(f"physical summary maximum fields differ in {key}")
        for name,scale in (("center_nearest_quadratic_eigenvalue_absolute",center_scale),
                           ("endpoint_nearest_eigenvalue_drift_absolute",drift_scale)):
            value=original["maxima"][name]
            derived=maxima[name]
            if (type(value) is not float or not math.isfinite(value) or value<0 or not math.isfinite(derived)
                    or abs(value-derived)>4*np.finfo(np.float64).eps*scale):
                raise ValueError(f"physical summary {key}/{name} exceeds the raw-matrix rounding allowance")
            maxima[name]=value
        # Canonical JSON distinguishes booleans, integers and floating values, including zero.
        canonical=lambda d: json.dumps(d,sort_keys=True,allow_nan=False,separators=(",",":"))
        if canonical(original)!=canonical(dict(observed,maxima=maxima)):
            raise ValueError(f"physical summary differs from individual physical receipts in {key}")


def package(args):
    roster,raw=historic()
    protocol=json.loads((HERE/"protocol.json").read_bytes())
    rows=[json.loads((args.out/"rows"/(key+".json")).read_bytes()) for key in sorted(roster)]
    if {p.stem for p in (args.out/"rows").glob("*.json")}!=roster.keys():
        raise ValueError("completion directory does not equal the 89 registered rows")
    from topology import run as measure_topology
    args.evidence.mkdir(parents=True,exist_ok=True)
    surface_file=args.evidence/"center_topology.json"
    measure_topology(argparse.Namespace(out=args.out,export=surface_file))
    surfaces=json.loads(surface_file.read_bytes())["rows"]
    for row in rows:
        include_topology(row,surfaces[row["row_id"]])
        check_row(row,roster[row["row_id"]],raw[row["row_id"]],protocol,source_check=True)
    runtime={a:json.loads((args.out/"geometries"/(a+".json")).read_bytes()) for a in ("maskB","combB","R01","R02")}
    data=dict(schema="compact-v21.physical-stratum-replay.receipts.v1",protocol=protocol,
              protocol_sha256=sha(HERE/"protocol.json"),runtime=runtime,rows=rows)
    args.evidence.mkdir(parents=True,exist_ok=True)
    path=args.evidence/NAME
    with NamedTemporaryFile(dir=args.evidence,delete=False) as temp:
        with gzip.GzipFile(filename="",mode="wb",fileobj=temp,mtime=0,compresslevel=9) as archive:
            archive.write(json.dumps(data,allow_nan=False,sort_keys=True,separators=(",",":")).encode())
        stage=Path(temp.name)
    stage.replace(path)
    summary=dict(schema="compact-v21.physical-stratum-replay.summary.v1", archive=NAME,
                 archive_sha256=sha(path), historical_scope_manifest_sha256=sha(NATIVE_EVIDENCE/"native_replay.json"),
                 registered_query_archive_sha256=sha(NATIVE_EVIDENCE/"native_targets.npz"),
                 protocol_sha256=data["protocol_sha256"], center_topology_sha256=sha(surface_file), groups=groups(rows),
                 exceptional_directions=exceptions(rows))
    (args.evidence/"summary.json").write_text(json.dumps(summary,sort_keys=True,indent=2,allow_nan=False)+"\n")
    verify(args)


def verify(args):
    roster,raw=historic()
    summary=json.loads((args.evidence/"summary.json").read_bytes())
    path=args.evidence/summary["archive"]
    if sha(path)!=summary["archive_sha256"] or sha(HERE/"protocol.json")!=summary["protocol_sha256"]:
        raise ValueError("receipt archive or physical protocol digest mismatch")
    if (sha(NATIVE_EVIDENCE/"native_replay.json")!=summary["historical_scope_manifest_sha256"] or
            sha(NATIVE_EVIDENCE/"native_targets.npz")!=summary["registered_query_archive_sha256"]):
        raise ValueError("the registered roster or original query archive changed")
    data=json.loads(gzip.decompress(path.read_bytes()))
    surface_file=args.evidence/"center_topology.json"
    if sha(surface_file)!=summary["center_topology_sha256"]:
        raise ValueError("actual component census archive changed")
    surfaces=json.loads(surface_file.read_bytes())["rows"]
    if data["protocol"]!=json.loads((HERE/"protocol.json").read_bytes()) or data["protocol_sha256"]!=summary["protocol_sha256"]:
        raise ValueError("portable embedded protocol differs")
    ids=[r["row_id"] for r in data["rows"]]
    if ids!=sorted(roster):
        raise ValueError("duplicate or missing registered row")
    for r in data["rows"]:
        runtime=data["runtime"][r["geometry_id"]]
        fp=r["run_fingerprint"]
        if (runtime["versioned_context"]!=r["context"] or runtime["registered_context"]!=r["legacy_context"]
                or runtime["physical_context"]!=r["registered_physics_context"] or runtime["geometry_hashes"]!=r["geometry_files"]
                or fp["integrator"]!=r["finite_difference_integrator"] or fp["protocol_sha256"]!=data["protocol_sha256"]
                or fp["replay_code_sha256"]!=sha(HERE/"replay_registered.py") or fp["checks_code_sha256"]!=sha(HERE/"checks.py")
                or (fp["integrator"]=="fast" and (fp["primal_code_sha256"]!=sha(HERE.parent/"primal.py")
                                                    or fp["primal_transform_sha256"]!=sha(HERE.parent/"primal_transform.cpp")))):
            raise ValueError("versioned runtime or measured physical replay sources differ")
        if r["material"].get("center_surface_components")!=surfaces[r["row_id"]]:
            raise ValueError("per-row topology evidence differs from the measured center census")
        check_row(r,roster[r["row_id"]],raw[r["row_id"]],data["protocol"])
    verify_summary_groups(summary["groups"], groups(data["rows"]), data["rows"])
    if summary["exceptional_directions"]!=exceptions(data["rows"]):
        raise ValueError("summary differs from individual physical receipts")
    print(json.dumps(summary["groups"]["all"],indent=2,ensure_ascii=False))


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--evidence",type=Path,default=HERE/"evidence")
    cmd=p.add_subparsers(dest="op",required=True)
    sub=cmd.add_parser("package");sub.add_argument("--out",type=Path,required=True)
    cmd.add_parser("verify")
    args=p.parse_args()
    package(args) if args.op=="package" else verify(args)
