"""Tests execute real C++ numerical code on declared manufactured maps.

Set CPINO_NATIVE_LIBRARY to a bridge built by build_native_solver_bridge.
Missing compiled dependencies are explicit SKIPs, never native pass receipts.
These fixtures are NOT historical queries or independent mesh validation.
"""
import copy
import hashlib
import json
import os
from pathlib import Path

import numpy as np
import pytest

from .endpoint_attribution import ORDER, COEFFICIENT_MULTIPLIERS, attribution, sha256
from .native_solver_adapter import (
    CONVENTION, PROFILE, STATE_SCHEMA, SYNTHETIC_GEOMETRY_PROFILE, NativeAdapterError, NativeKernel,
    NativeSolverAdapter, _packet, _unit_coefficients, factory,
)


THRESHOLDS = {"stationarity": 1e-12, "balance": 1e-6}
TOPOLOGY = hashlib.sha256(b"manufactured-map-one-stratum").hexdigest()


class ManufacturedMap:
    """Affine numerical fixture: no voxel, native Boolean or training claim."""
    def __init__(self, runtime_path=None):
        self.runtime_path = runtime_path

    def __call__(self, c):
        m = np.array([1., .02, -.01, .1, .3, .4, .5, 0., 0., 0.]) + .1 * c
        identity=hashlib.sha256(np.asarray(c, dtype="<f8").tobytes()).hexdigest()
        return dict(raw_moments=m, moment_jacobian=.1*np.eye(10), derivative_defined=True,
                    geometry_hash=identity,topology_hash=TOPOLOGY,
                    geometry_map_profile=SYNTHETIC_GEOMETRY_PROFILE,
                    evaluation_map_identity=identity,source_scientific_hash=identity,
                    boolean_backend_identity="manufactured_python_affine_v1",derivative_identity="manufactured_affine_derivative_v1")


def manufactured_factory(case, state):
    return ManufacturedMap(state["trial_evaluator"]["runtime_artifact"]["path"])


@pytest.fixture(scope="session")
def kernel_path():
    value = os.environ.get("CPINO_NATIVE_LIBRARY")
    if not value:
        pytest.skip("compiled native bridge unavailable; set CPINO_NATIVE_LIBRARY (no native proof)")
    path = Path(value).resolve()
    assert path.is_file(), "an explicitly configured native bridge must exist"
    return path


@pytest.fixture
def kernel(kernel_path):
    return NativeKernel(kernel_path, objective={"mode": "STATIC_MAXIMUM_MASS"},
                        thresholds=THRESHOLDS, max_evaluations=10)


def query():
    c=np.array([.8, .13, -.09, .2, .1, -.07, .06, .03, -.02, .04])
    return c/np.linalg.norm(c)


def make_case(tmp_path, kernel_path, kernel):
    c=query()
    native=ManufacturedMap()(c)
    meta=dict(geometry_id="manufactured_affine_map", query_id="q0", source_group_id="fixture_only",
              split="synthetic", teacher_version="manufactured_linear_map_v1", solver_version=PROFILE,
              teacher_map_profile=SYNTHETIC_GEOMETRY_PROFILE,
              native_geometry_backend_identity="manufactured_python_affine_v1",
              native_solver_profile=PROFILE, coordinate_frame="fixture_xyz", units="synthetic_unit",
              trajectory_stage="iteration", chart_id="raw_unit", moment_order=ORDER,
              coefficient_basis="raw_coefficients", coefficient_multipliers=COEFFICIENT_MULTIPLIERS,
              thresholds=THRESHOLDS.copy(), editable_mass=1., length_scale=1., coefficient_scale=1.,
              derivative_valid=True, derivative_dimension=10,
              adapter_contract=dict(acceptance_reads_jacobian=False, multiplier_policy="native_recompute"),
              replay_tolerances=dict(residual_atol=1e-14, residual_rtol=1e-12, direction_atol=1e-14,
                                     direction_rtol=1e-12, coefficient_atol=1e-14, coefficient_rtol=1e-12),
              artifacts={})
    for role in ("geometry","operand","manifest","checkpoint"):
        path=tmp_path/(role+".fixture")
        path.write_text("manufactured fixture only: "+role+"\n")
        meta[role+"_sha256"]=sha256(path)
        meta["artifacts"][role]={"path":str(path)}
    meta["solver_sha256"]=sha256(kernel_path)
    meta["artifacts"]["solver"]={"path":str(kernel_path)}
    receipt=kernel_path.with_suffix(kernel_path.suffix+".build.json")
    state=dict(schema=STATE_SCHEMA, profile=PROFILE, equation_profile="DUAL_MONOMIAL_RAW_V1",
               coefficient_convention=CONVENTION, external_state_kind="repository_stateless_spherical_newton_v1",
               geometry_map_profile=SYNTHETIC_GEOMETRY_PROFILE,boolean_backend_identity="manufactured_python_affine_v1",
               multiplier_policy="native_recompute", thresholds=THRESHOLDS.copy(),
               objective={"mode":"STATIC_MAXIMUM_MASS"},
               solver_settings=dict(maximum_iterations=1,maximum_restarts=1,line_search_reduction=.5,
                                    maximum_geodesic_step=.5,maximum_function_evaluations_per_restart=10),
               bindings={k:meta[k] for k in ("geometry_sha256","operand_sha256","teacher_version","coordinate_frame","units")},
               center_geometry_hash=native["geometry_hash"],center_topology_hash=TOPOLOGY,
               center_evaluation_map_identity=native["evaluation_map_identity"],
               kernel_build_receipt={"path":str(receipt),"sha256":sha256(receipt)},
               trial_evaluator={"factory":__name__+":manufactured_factory","source_sha256":sha256(__file__),
                                "runtime_artifact":{"path":str(kernel_path),"sha256":sha256(kernel_path)}})
    state_path=tmp_path/"state.json"
    state_path.write_text(json.dumps(state,sort_keys=True)+"\n")
    meta["artifacts"]["solver_state"]={"path":str(state_path)}
    meta["adapter_contract"]["external_state_sha256"]=sha256(state_path)
    m=native["raw_moments"]
    j=native["moment_jacobian"]
    case=dict(meta=meta,asset_root=str(tmp_path),coefficients=c,native_moments=m,native_jacobian=j,
              predicted_moments=m+np.array([.01,.002,-.001,.01,0,0,0,0,0,0]),
              predicted_jacobian=j*1.2,chart_basis=np.eye(10),direction=np.ones(10))
    meta["native_replay"]={"accept":kernel.accept(c,m),
                           "step":kernel.step(c,m,j,ManufacturedMap(),geometry_hash=native["geometry_hash"],topology_hash=TOPOLOGY)}
    return case


def test_unit_coefficients_and_derivative_null_are_strict_without_a_runtime():
    with pytest.raises(NativeAdapterError,match="no silent normalization"):
        _unit_coefficients(np.ones(10))
    with pytest.raises(NativeAdapterError,match="undefined derivative must be null"):
        _packet(np.ones(10),np.zeros((10,10)),geometry_hash="a"*64,topology_hash="b"*64,defined=False)
    # True zero derivatives are valid data, even if this native optimizer
    # later rejects a zero-sensitivity initial state.
    assert _packet(np.ones(10),np.zeros((10,10)),geometry_hash="a"*64,topology_hash="b"*64).derivative_defined == 1


def test_native_acceptance_ignores_j_and_recomputes_multipliers(kernel_path):
    k=NativeKernel(kernel_path,objective={"mode":"STATIC_LOWER_CENTER"},thresholds=THRESHOLDS)
    c=query()
    m=ManufacturedMap()(c)["raw_moments"]
    first=k.accept(c,m,None)
    assert first==k.accept(c,m,np.full((10,10),np.nan))
    changed=m.copy();changed[3]+=.2
    second=k.accept(c,changed,None)
    assert first["multiplier_mu"] != second["multiplier_mu"]
    assert first["stationarity_residual"] != second["stationarity_residual"]
    # Independent analytic first derivatives for z/m; no cut Jacobian here.
    grad=np.zeros(10);grad[0]=-m[3]/m[0]**2;grad[3]=1/m[0]
    dual=c*np.array([1]*7+[2]*3);dual/=np.linalg.norm(dual)
    projection=np.column_stack((-np.eye(10)[:,1:3],dual))
    coefficients=np.linalg.lstsq(projection,-grad,rcond=None)[0]
    residual=grad+projection@coefficients
    expected=np.sqrt(residual@residual+m[1]**2+m[2]**2)
    assert first["stationarity_residual"] == pytest.approx(expected,rel=1e-12)
    np.testing.assert_allclose(first["multipliers"],coefficients[:2],rtol=1e-12,atol=1e-14)


def test_native_acceptance_includes_positive_multiplier_requirement(kernel):
    c=np.eye(10)[0]
    m=np.array([1.,0.,0.,.1,.3,.4,.5,0.,0.,0.])
    assert kernel.accept(c,m)["accepted"] is True
    reverse=kernel.accept(-c,m)
    assert reverse["stationarity_residual"] < 1e-12
    assert reverse["multiplier_mu"] < 0 and reverse["accepted"] is False


def test_spinning_top_native_objective_and_moment_order(kernel_path):
    weight=.37
    k=NativeKernel(kernel_path,objective={"mode":"SPINNING_TOP","weight":weight},thresholds=THRESHOLDS)
    c=query();m=ManufacturedMap()(c)["raw_moments"]
    actual=k.accept(c,m)
    assert actual==k.accept(c,m,np.full((10,10),np.nan))
    assert np.isfinite([actual["objective"],actual["stationarity_residual"],actual["balance_residual"]]).all()
    i11=m[5]+m[6]-(m[2]**2+m[3]**2)/m[0]
    i22=m[4]+m[6]-(m[1]**2+m[3]**2)/m[0]
    i33=m[4]+m[5]-(m[1]**2+m[2]**2)/m[0]
    i12=-m[9]+m[1]*m[2]/m[0]
    expected=(i11**2+i22**2+2*i12**2)/i33**2+weight*m[3]/m[0]
    assert actual["objective"]==pytest.approx(expected,rel=1e-13)
    np.testing.assert_allclose(actual["constraints"],
        [m[1],m[2],-m[8]+m[1]*m[3]/m[0],-m[7]+m[2]*m[3]/m[0]],rtol=1e-13)
    changed=m.copy();changed[3]+=.02
    assert k.accept(c,changed)["stationarity_residual"] != actual["stationarity_residual"]
    assert k.accept(c,changed)["multiplier_mu"] != actual["multiplier_mu"]
    pure=NativeKernel(kernel_path,objective={"mode":"SPINNING_TOP_LOWER_CENTER","weight":0.},thresholds=THRESHOLDS)
    height=pure.accept(c,m)
    assert height==pure.accept(c,m,np.full((10,10),np.nan))
    assert height["objective"]==pytest.approx(m[3]/m[0],rel=1e-13)
    assert np.isfinite([height["objective"],height["stationarity_residual"],height["balance_residual"]]).all()
    assert pure.accept(c,changed)["multiplier_mu"] != height["multiplier_mu"]


def test_actual_native_step_and_fresh_trials_are_reproducible(kernel):
    c=query();native=ManufacturedMap()(c)
    kwargs=dict(geometry_hash=native["geometry_hash"],topology_hash=TOPOLOGY)
    first=kernel.step(c,native["raw_moments"],native["moment_jacobian"],ManufacturedMap(),**kwargs)
    second=kernel.step(c,native["raw_moments"],native["moment_jacobian"],ManufacturedMap(),**kwargs)
    assert first==second
    assert first["line_search_accepted"] is True
    assert first["trial_evaluations"]>=1
    assert first["function_evaluations"]==1+first["trial_evaluations"]
    assert first["direction_origin"]=="native_event_bordered_system_replay"
    assert any(event["type"]=="newton_linear_solve" for event in first["events"])
    trial=np.asarray(first["trial_coefficients"])
    actual=kernel.accept(trial,ManufacturedMap()(trial)["raw_moments"])
    assert first["trial_native_stationarity_residual"]==actual["stationarity_residual"]
    assert actual["stationarity_residual"]<kernel.accept(c,native["raw_moments"])["stationarity_residual"]


def test_valid_zero_j_and_unavailable_trial_derivatives_are_different(kernel):
    c=query();native=ManufacturedMap()(c)
    kwargs=dict(geometry_hash=native["geometry_hash"],topology_hash=TOPOLOGY)
    zero=kernel.step(c,native["raw_moments"],np.zeros((10,10)),ManufacturedMap(),**kwargs)
    assert zero["reason"]=="initial_sensitivity_vanished"
    assert zero["trial_evaluations"]==0

    def invalid(trial):
        packet=ManufacturedMap()(trial)
        packet.update(derivative_defined=False,moment_jacobian=None,derivative_diagnostic="event neighborhood unavailable")
        return packet

    rejected=kernel.step(c,native["raw_moments"],native["moment_jacobian"],invalid,**kwargs)
    assert rejected["line_search_accepted"] is False
    assert rejected["trial_evaluations"]==kernel.max_evaluations-1
    assert all(row["derivative_defined"] is False for row in rejected["trial_receipts"])
    assert any("derivative is undefined" in row["detail"] for row in rejected["events"])


def test_bound_four_arm_replay_and_m_only_invalid_j(kernel_path,kernel,tmp_path):
    case=make_case(tmp_path,kernel_path,kernel)
    report=attribution(case,factory,run_step=True)
    assert report["endpoint_J_invariance_observed"] is True
    assert report["fixed_endpoint_repeat_identical"] is True
    assert report["one_step"]["D_nativeM_nativeJ"]["line_search_accepted"] is True
    assert report["status"]=="DIAGNOSTIC_ONLY_NO_TRAINING_PROMOTION"
    reduced=copy.deepcopy(case)
    reduced["meta"].update(derivative_valid=False,derivative_reason="event neighborhood unavailable")
    for key in ("native_jacobian","predicted_jacobian","chart_basis","direction"):
        reduced.pop(key)
    assert attribution(reduced,factory)["derivative_status"]=="unavailable_no_jacobian_claim"


def test_incompatible_profile_stale_artifact_and_reduced_step_reject(kernel_path,kernel,tmp_path):
    case=make_case(tmp_path,kernel_path,kernel)
    wrong=copy.deepcopy(case);wrong["meta"]["native_solver_profile"]="PRIVATE_1756"
    with pytest.raises(NativeAdapterError,match="incompatible solver profile"):
        factory(wrong)
    wrong_teacher=copy.deepcopy(case)
    wrong_teacher["meta"]["teacher_map_profile"]="DEPLOYED_CHERCHI_FINITE_SOURCE_V1"
    with pytest.raises(NativeAdapterError,match="CGAL cannot substitute for deployed Cherchi"):
        factory(wrong_teacher)
    reduced=copy.deepcopy(case);reduced["meta"]["derivative_dimension"]=8
    reduced["chart_basis"]=np.eye(10)[:,:8]
    with pytest.raises(NativeAdapterError,match="never zero-pad"):
        factory(reduced).step(reduced["coefficients"],reduced["native_moments"],reduced["native_jacobian"][:,:8])
    adapter=factory(case)
    adapter.accept(case["coefficients"],case["native_moments"],None)
    with pytest.raises(NativeAdapterError,match="fresh native adapter"):
        adapter.accept(case["coefficients"],case["native_moments"],None)
    Path(case["meta"]["artifacts"]["geometry"]["path"]).write_text("changed")
    with pytest.raises(ValueError,match="checksum mismatch|artifact hash mismatch|digest mismatch"):
        factory(case)


def write_smoke_receipt(library, output):
    """Persist real kernel outputs with explicit numerical-fixture ceilings."""
    import zipfile
    library=Path(library).resolve()
    c=query();packet=ManufacturedMap()(c)
    runs={}
    for mode,weight in (("STATIC_MAXIMUM_MASS",0.),("SPINNING_TOP",.37),("SPINNING_TOP_LOWER_CENTER",0.)):
        kernel=NativeKernel(library,objective={"mode":mode,"weight":weight},thresholds=THRESHOLDS,max_evaluations=10)
        result=kernel.accept(c,packet["raw_moments"])
        assert result==kernel.accept(c,packet["raw_moments"],np.full((10,10),np.nan))
        changed=packet["raw_moments"].copy();changed[3]+=.02
        runs[mode]=dict(objective={"mode":mode,"weight":weight},accept=result,same_M_any_J_identical=True,
                        changed_z_accept=kernel.accept(c,changed))
        if mode=="STATIC_MAXIMUM_MASS":
            step=kernel.step(c,packet["raw_moments"],packet["moment_jacobian"],ManufacturedMap(),
                             geometry_hash=packet["geometry_hash"],topology_hash=TOPOLOGY)
            assert step["line_search_accepted"] is True
            runs[mode]["step"]=step
    root=Path(__file__).resolve().parents[3]
    archive=root/"experiments/compact_pino_v21/stratum/qualification/native_admission/evidence/replay_inputs.zip"
    archived=[]
    kernel=NativeKernel(library,objective={"mode":"STATIC_MAXIMUM_MASS"},thresholds=THRESHOLDS)
    with zipfile.ZipFile(archive) as source:
        for member in ("centers/R02_R8_032.json","centers/maskB_R8_035.json"):
            data=source.read(member);record=json.loads(data)
            assert record["material"]["moment_valid"] is True
            result=kernel.accept(record["query"],record["material"]["moments"])
            archived.append(dict(member=member,member_sha256=hashlib.sha256(data).hexdigest(),row_id=record["row_id"],
                                 input_kind="archived_M_and_unit_query_only_no_live_mesh",
                                 selected_diagnostic_objective="STATIC_MAXIMUM_MASS",native_formula_output=result,
                                 historical_acceptance_reproduced=False,derivative_qualification_upgraded=False))
    receipt=dict(schema="cpino_native_kernel_execution_receipt_v1",profile=PROFILE,kernel_path=str(library),
                 kernel_sha256=sha256(library),build_receipt_path=str(library)+".build.json",
                 build_receipt_sha256=sha256(str(library)+".build.json"),
                 fixture_source_sha256=sha256(__file__),adapter_source_sha256=sha256(Path(__file__).with_name("native_solver_adapter.py")),
                 manufactured_fixture=dict(coefficients=c.tolist(),moments=packet["raw_moments"].tolist(),
                                           jacobian=packet["moment_jacobian"].tolist(),
                                           claim_scope="actual_repository_cpp_numeric_kernel_on_manufactured_map"),
                 manufactured_runs=runs,archive_sha256=sha256(archive),archived_formula_inputs=archived,
                 private_1756_solver_reproduced=False,live_native_mesh_reproduced=False,
                 deployed_Cherchi_teacher_connected=False,full_or_reduced_teacher_admission_granted=False,
                 long_training_authorized=False,promotion_status="NOT_EVALUATED_CURRENT_PILOT_REMAINS_NO_GO")
    output=Path(output)
    with output.open("x") as stream:
        json.dump(receipt,stream,indent=2,allow_nan=False);stream.write("\n")
    return receipt


if __name__=="__main__":
    import argparse
    parser=argparse.ArgumentParser(description="Actual native numerical smoke; not a physical acceptance run")
    parser.add_argument("--smoke-library",required=True,type=Path)
    parser.add_argument("--smoke-output",required=True,type=Path)
    args=parser.parse_args()
    value=write_smoke_receipt(args.smoke_library,args.smoke_output)
    print(json.dumps({key:value[key] for key in ("schema","kernel_sha256","private_1756_solver_reproduced","live_native_mesh_reproduced","promotion_status")},indent=2))
