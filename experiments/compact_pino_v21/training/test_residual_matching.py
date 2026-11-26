"""Numerical/gradient contracts, not a model-quality or deployment receipt.

Native oracle (optional only when Eigen/compiler are genuinely unavailable)::

    CPINO_EIGEN_ROOT=/absolute/eigen python -m pytest -q \
        experiments/compact_pino_v21/training/test_residual_matching.py

The fixture compiles the repository's UNMODIFIED SpinMomentProblem.cpp. The
small C++ harness below only chooses its public mode and serializes results;
it contains no replacement objective, constraint, gradient or Hessian formula.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import shutil
import subprocess
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from experiments.compact_pino_v21.training.losses import physical_loss
from experiments.compact_pino_v21.training.residual_matching import (
    TARGET_SCHEMA, REPO_PROFILE, LINEARIZED_PROFILE, P_CONVENTION,
    MOMENT_ORDER, REPO_FORMULA_SHA256, PhysicalDomainError, repository_objective_constraints,
    validate_residual_target, residual_match_terms, residual_metrics,
)


NATIVE = np.array([1.7, .12, -.08, .31, .63, .72, .58, -.04, .03, .09])
DELTA = np.array([.003, -.001, .0007, -.002, .0003, -.0008, .0011, .0002, -.0003, .0004])
COEFFICIENTS = np.array([.8, .13, -.09, .2, .1, -.07, .06, .03, -.02, .04])
DUAL = COEFFICIENTS * np.array([1.] * 7 + [2.] * 3)
DUAL /= np.linalg.norm(DUAL)
MODES = ("STATIC_MAXIMUM_MASS", "STATIC_LOWER_CENTER", "STATIC_PARETO", "FIDGET_SPINNER",
         "YOYO", "SPINNING_TOP", "SPINNING_TOP_LOWER_CENTER")
COUNTS = (2,2,2,2,5,4,4)


def exact_target(mode="SPINNING_TOP"):
    count = COUNTS[MODES.index(mode)]
    return dict(schema=TARGET_SCHEMA, valid=True, profile=REPO_PROFILE,
        teacher_role="synthetic_fixture", moment_order=list(MOMENT_ORDER),
        formula_provenance=dict(source_profile=REPO_PROFILE, source_sha256=dict(REPO_FORMULA_SHA256)),
        objective=dict(mode=mode,weight=0. if mode == "SPINNING_TOP_LOWER_CENTER" else .37,mass_scale=1.31),
        reference_moments=NATIVE.tolist(),p=DUAL.tolist(),p_convention=P_CONVENTION,
        multipliers=np.linspace(-.17,.23,count).tolist(),multiplier_mu=.43,
        stationarity_scale=np.linspace(.8,1.7,10).tolist(),constraint_scale=np.linspace(.4,.9,count).tolist())


def linearized_target():
    target = exact_target()
    target.update(profile=LINEARIZED_PROFILE,
        formula_provenance=dict(source_profile="EXPLICIT_NATIVE_SENSITIVITY_FIXTURE",
                                source_sha256={"native_fixture_formula": "d"*64}),
        lagrangian_hessian=(np.diag(np.arange(1.,11.))-2.4*np.eye(10)).tolist(),
        constraint_jacobian=np.eye(10)[:4].tolist(),
        reference_stationarity=np.linspace(.5,1.4,10).tolist(),
        reference_constraints=[.2,.4,-.1,.3])
    del target["objective"]
    return target


def _record(target, *, moment_weight=1., residual_weight=1.):
    row = dict(moment_weight=moment_weight,jacobian_weight=1.,residual_weight=residual_weight,
               case=dict(meta=dict(editable_mass=1.,length_scale=1.,derivative_valid=False),
                         native_moments=NATIVE.copy(),coefficients=COEFFICIENTS.copy()))
    if target is not None:
        row["residual_target"] = target
    return row


def _prediction(delta=DELTA, *, requires_grad=True):
    return SimpleNamespace(moments=torch.tensor(NATIVE+delta,dtype=torch.float64,requires_grad=requires_grad),
                           jacobian=None)


def _numpy_objective_constraints(m, objective):
    """Independent full 3x3 covariance/inertia construction for FD checks."""
    mass, first = m[0], m[1:4]
    second = np.array([[m[4],m[9],m[8]], [m[9],m[5],m[7]], [m[8],m[7],m[6]]])
    central = second - np.outer(first,first)/mass
    inertia = np.trace(central)*np.eye(3)-central
    mode, weight, scale = objective["mode"], objective["weight"], objective["mass_scale"]
    horizontal = first[:2]
    if mode == "STATIC_MAXIMUM_MASS":
        return -scale*mass,horizontal
    if mode == "STATIC_LOWER_CENTER":
        return first[2]/mass,horizontal
    if mode == "STATIC_PARETO":
        return -weight*scale*mass+(1-weight)*first[2]/mass,horizontal
    if mode == "FIDGET_SPINNER":
        return -(m[4]+m[5]),horizontal
    constraints = np.r_[first if mode == "YOYO" else horizontal,inertia[0,2],inertia[1,2]]
    if mode == "SPINNING_TOP_LOWER_CENTER":
        return first[2]/mass,constraints
    value = np.sum(inertia[:2,:2]**2)/inertia[2,2]**2
    return value if mode == "YOYO" else value+weight*first[2]/mass,constraints


def _fd_gradient(fn, point, step=2e-4):
    columns = np.eye(len(point))*step
    return np.array([(fn(point-2*d)-8*fn(point-d)+8*fn(point+d)-fn(point+2*d))/(12*step)
                     for d in columns])


@pytest.mark.parametrize("mode",MODES)
def test_exact_profile_matches_independent_inertia_and_finite_difference(mode):
    target = exact_target(mode)
    m = torch.tensor(NATIVE+DELTA,dtype=torch.float64,requires_grad=True)
    objective,constraints = repository_objective_constraints(m,target["objective"])
    expected_f,expected_c = _numpy_objective_constraints(NATIVE+DELTA,target["objective"])
    assert float(objective.detach()) == pytest.approx(expected_f,rel=2e-14,abs=2e-14)
    np.testing.assert_allclose(constraints.detach(),expected_c,rtol=2e-14,atol=2e-14)
    def lagrangian(values):
        f,c = _numpy_objective_constraints(values,target["objective"])
        return f-np.dot(c,target["multipliers"])
    expected_gradient = _fd_gradient(lagrangian,NATIVE+DELTA)
    terms = residual_match_terms(m,NATIVE,target,coefficients=COEFFICIENTS)
    expected_r = expected_gradient+target["multiplier_mu"]*DUAL
    np.testing.assert_allclose(terms["predicted_stationarity"].detach(),expected_r,rtol=2e-9,atol=2e-10)


def test_nonzero_reference_is_not_a_zero_training_target():
    target = exact_target()
    prediction = _prediction(np.zeros(10))
    terms = residual_match_terms(prediction.moments,NATIVE,target,coefficients=COEFFICIENTS)
    assert torch.linalg.vector_norm(terms["reference_stationarity"]) > 1.
    assert torch.linalg.vector_norm(terms["reference_constraints"]) > .1
    loss,info = physical_loss([prediction],[_record(target)],residual_loss_weight=3.,constraint_loss_weight=2.)
    assert float(loss.detach()) == 0.
    assert info["residual_mask"] == info["constraint_mask"] == [True]
    loss.backward()
    assert torch.equal(prediction.moments.grad,torch.zeros(10,dtype=torch.float64))


def test_second_derivative_chain_passes_gradcheck_and_teachers_stay_frozen():
    target = exact_target()
    teacher = torch.tensor(NATIVE,dtype=torch.float64,requires_grad=True)
    for key in ("p","multipliers","multiplier_mu","reference_moments","stationarity_scale","constraint_scale"):
        target[key] = torch.tensor(target[key],dtype=torch.float64,requires_grad=True)
    target["objective"]["weight"] = torch.tensor(.37,dtype=torch.float64,requires_grad=True)
    prediction = torch.tensor(NATIVE+DELTA,dtype=torch.float64,requires_grad=True)
    def function(m):
        terms = residual_match_terms(m,teacher,target,coefficients=COEFFICIENTS)
        return terms["stationarity_scaled_delta"].square().sum()+terms["constraint_scaled_delta"].square().sum()
    assert torch.autograd.gradcheck(function,(prediction,),eps=1e-6,atol=2e-6,rtol=2e-4)
    function(prediction).backward()
    assert prediction.grad is not None and torch.isfinite(prediction.grad).all() and prediction.grad.abs().sum() > 0
    assert teacher.grad is None
    assert all(target[key].grad is None for key in ("p","multipliers","multiplier_mu","reference_moments",
                                                   "stationarity_scale","constraint_scale"))
    assert target["objective"]["weight"].grad is None


def test_fixed_mu_cancels_analytically_and_no_graph_is_invented():
    target = exact_target()
    expected = residual_match_terms(torch.tensor(NATIVE+DELTA),NATIVE,target)
    target["multiplier_mu"] = 1e30
    actual = residual_match_terms(torch.tensor(NATIVE+DELTA),NATIVE,target)
    assert torch.equal(actual["stationarity_delta"],expected["stationarity_delta"])
    assert not actual["stationarity_delta"].requires_grad
    loss,_ = physical_loss([_prediction(requires_grad=False)],[_record(target)],residual_loss_weight=1.)
    assert not loss.requires_grad


def test_full_vector_linearization_preserves_cancelling_m_z_zz_terms():
    target = linearized_target()
    direction = np.zeros(10); direction[[0,3,6]] = [1.,2.,-3.]
    hessian = np.outer(direction,direction)
    target["lagrangian_hessian"] = hessian.tolist()
    target["constraint_jacobian"] = np.zeros((4,10)).tolist()
    delta = np.zeros(10); delta[[0,3,6]] = [.01,.01,.01]
    prediction = torch.tensor(NATIVE+delta,dtype=torch.float64,requires_grad=True)
    terms = residual_match_terms(prediction,NATIVE,target)
    assert np.linalg.norm(hessian[:,0]*delta[0]) > 0.
    assert np.linalg.norm(hessian[:,3]*delta[3]) > 0.
    assert np.linalg.norm(hessian[:,6]*delta[6]) > 0.
    assert torch.linalg.vector_norm(terms["stationarity_delta"]) < 1e-15
    assert terms["approximation"] == "first_order_at_native_moments"
    assert torch.linalg.vector_norm(terms["reference_stationarity"]) > 1.
    info = residual_metrics(prediction,NATIVE,target,coefficients=COEFFICIENTS)
    assert info["approximation"] == "first_order_at_native_moments"
    assert "accepted" not in info and "exact_stationarity" not in info


def test_linearized_native_matrices_scales_and_references_are_frozen():
    target = linearized_target()
    for key in ("lagrangian_hessian","constraint_jacobian","reference_stationarity","reference_constraints",
                "stationarity_scale","constraint_scale"):
        target[key] = torch.tensor(target[key],dtype=torch.float64,requires_grad=True)
    theta = torch.tensor(.2,dtype=torch.float64,requires_grad=True)
    moments = torch.tensor(NATIVE,dtype=torch.float64)+torch.tensor(DELTA,dtype=torch.float64)*torch.sigmoid(theta)
    terms = residual_match_terms(moments,NATIVE,target)
    loss = terms["stationarity_scaled_delta"].square().sum()+terms["constraint_scaled_delta"].square().sum()
    loss.backward()
    assert theta.grad is not None and torch.isfinite(theta.grad) and theta.grad != 0
    assert all(target[k].grad is None for k in ("lagrangian_hessian","constraint_jacobian","reference_stationarity",
                                               "reference_constraints","stationarity_scale","constraint_scale"))


def test_independent_physical_denominators_and_missing_jacobian_membership():
    targets = [exact_target(),None,dict(valid=False,reason="native replay unavailable"),exact_target()]
    rows = [_record(t,moment_weight=m,residual_weight=w) for t,m,w in zip(targets,[1.,4.,1.,0.],[2.,10.,20.,1.])]
    predictions = [_prediction(DELTA*i) for i in (1,2,3,4)]
    loss,info = physical_loss(predictions,rows,residual_loss_weight=2.,constraint_loss_weight=.5)
    assert info["moment_denominator"] == 6.
    assert info["jacobian_denominator"] == 0.
    assert info["residual_denominator"] == info["constraint_denominator"] == 3.
    assert info["residual_mask"] == info["constraint_mask"] == [True,False,False,True]
    assert info["residual_weight"] == pytest.approx([4/3,0.,0.,2/3])
    assert info["constraint_weight"] == pytest.approx([1/3,0.,0.,1/6])
    assert float(loss.detach()) == pytest.approx(info["moment_loss"]+2*info["residual_loss"]+.5*info["constraint_loss"])
    loss.backward()
    assert all(p.moments.grad is not None and torch.isfinite(p.moments.grad).all() for p in predictions)


def test_random_batch_without_physics_remains_moment_only_with_zero_masks():
    row = _record(None)
    prediction = _prediction()
    old,old_info = physical_loss([prediction],[row])
    loss,info = physical_loss([prediction],[row],residual_loss_weight=1.,constraint_loss_weight=1.)
    assert torch.equal(loss,old)
    for name in ("residual","constraint"):
        assert info[name+"_denominator"] == 0.
        assert info[name+"_mask"] == [False]
        assert info[name+"_weight"] == [0.]
        assert info[name+"_loss"] == 0.
    assert info["moment_weight"] == old_info["moment_weight"]


def test_disabled_physics_keeps_previous_loss_and_does_not_read_targets():
    row = _record({"deliberately":"not a physical target"})
    prediction = _prediction()
    loss,info = physical_loss([prediction],[row])
    assert float(loss.detach()) == pytest.approx(np.mean(DELTA**2))
    assert info["residual_reasons"] == ["ZERO_LOSS_WEIGHT"]


def test_numpy_diagnostics_are_jsonable_and_do_not_touch_prediction_gradients():
    prediction = _prediction()
    result = residual_metrics(prediction.moments,NATIVE,exact_target(),coefficients=COEFFICIENTS)
    json.dumps(result,allow_nan=False)
    assert result["availability"] == "AVAILABLE"
    assert result["stationarity_error_l2"] > 0
    assert result["approximation"] == "exact_fixed_multiplier_residual_difference"
    assert prediction.moments.grad is None
    assert residual_metrics(NATIVE,NATIVE,None)["availability"] == "UNAVAILABLE_PROFILE"
    unavailable = dict(schema=TARGET_SCHEMA,valid=False,reason="no fixed native multipliers")
    assert residual_metrics(NATIVE,NATIVE,unavailable)["availability"] == "UNAVAILABLE_TARGET"


@pytest.mark.parametrize("field,value,match",[
    ("profile","PRIVATE_NOODLES_SOLVE_PY","unknown residual profile"),
    ("p_convention","raw_geometry","unsupported p_convention"),
    ("stationarity_scale",[0.]*10,"strictly positive"),
    ("constraint_scale",[1.]*3,"shape"),
    ("multipliers",[0.]*3,"shape"),
    ("p",[0.]*10,"already be unit"),
    ("reference_moments",(NATIVE+DELTA).tolist(),"case native teacher"),
    ("teacher_role",None,"teacher_role"),
])
def test_malformed_or_unbound_physics_is_rejected(field,value,match):
    target = exact_target(); target[field] = value
    with pytest.raises(ValueError,match=match):
        validate_residual_target(target,NATIVE,coefficients=COEFFICIENTS)


def test_query_and_source_binding_are_not_inferred_or_relabeled():
    with pytest.raises(ValueError,match="supplied query"):
        validate_residual_target(exact_target(),NATIVE,coefficients=-COEFFICIENTS)
    target = exact_target(); target["formula_provenance"]["source_profile"] = "PRIVATE_NOODLES"
    with pytest.raises(ValueError,match="relabelled"):
        validate_residual_target(target,NATIVE)
    target = exact_target(); target["formula_provenance"]["source_sha256"]["src/spin/SpinMomentProblem.cpp"] = "0"*64
    with pytest.raises(ValueError,match="source digest mismatch"):
        validate_residual_target(target,NATIVE)
    target = exact_target(); del target["objective"]["weight"]
    with pytest.raises(ValueError,match="must be explicit"):
        validate_residual_target(target,NATIVE)


def test_native_linearization_does_not_silently_repair_a_wrong_hessian():
    target = linearized_target(); target["lagrangian_hessian"][0][1] = 1.
    with pytest.raises(ValueError,match="symmetric"):
        validate_residual_target(target,NATIVE)


def test_pure_height_profile_does_not_evaluate_the_removed_inertia_ratio():
    moments = torch.tensor([1.,0.,0.,.2,0.,0.,.3,0.,0.,0.],dtype=torch.float64)
    value,constraints = repository_objective_constraints(moments,exact_target("SPINNING_TOP_LOWER_CENTER")["objective"])
    assert float(value) == .2 and len(constraints) == 4
    with pytest.raises(ValueError,match="axial central inertia denominator"):
        repository_objective_constraints(moments,exact_target("SPINNING_TOP")["objective"])


@pytest.mark.parametrize("prediction,reason",[
    (np.r_[0.,NATIVE[1:]],"positive mass"),
    (np.array([1.,0.,0.,.2,0.,0.,.3,0.,0.,0.]),"axial central inertia"),
])
def test_prediction_domain_failure_is_unavailable_for_metrics_but_stops_training(prediction,reason):
    target = exact_target()
    result = residual_metrics(prediction,NATIVE,target,coefficients=COEFFICIENTS)
    assert result["availability"] == "UNAVAILABLE_PREDICTION_DOMAIN"
    assert reason in result["reason"]
    assert result["profile"] == REPO_PROFILE
    assert "stationarity_error_l2" not in result and "stationarity_delta" not in result
    assert "constraint_error_l2" not in result and "accepted" not in result
    json.dumps(result,allow_nan=False)
    value = SimpleNamespace(moments=torch.tensor(prediction,dtype=torch.float64,requires_grad=True),jacobian=None)
    with pytest.raises(PhysicalDomainError,match=reason):
        physical_loss([value],[_record(target)],residual_loss_weight=1.,constraint_loss_weight=1.)


def test_invalid_native_reference_domain_is_never_downgraded_to_prediction_unavailability():
    invalid_reference = np.array([1.,0.,0.,.2,0.,0.,.3,0.,0.,0.])
    target = exact_target(); target["reference_moments"] = invalid_reference.tolist()
    with pytest.raises(ValueError,match="invalid native reference physical domain"):
        validate_residual_target(target,invalid_reference,coefficients=COEFFICIENTS)
    with pytest.raises(ValueError,match="invalid native reference physical domain"):
        residual_metrics(NATIVE,invalid_reference,target,coefficients=COEFFICIENTS)


@pytest.mark.parametrize("failure",["profile","query"])
def test_bad_profile_or_query_binding_raises_before_invalid_prediction_domain(failure):
    target = exact_target()
    coefficients = COEFFICIENTS
    if failure == "profile":
        target["profile"] = "UNVERIFIED_PRIVATE_FORMULA"
        match = "unknown residual profile"
    else:
        coefficients = -COEFFICIENTS
        match = "supplied query"
    with pytest.raises(ValueError,match=match):
        residual_metrics(np.r_[0.,NATIVE[1:]],NATIVE,target,coefficients=coefficients)


_CPP_ORACLE = r'''
#include "spin/SpinMomentProblem.h"
#include <iomanip>
#include <iostream>
#include <stdexcept>
spin::SpinMomentProblem select(int mode,double weight,double scale) {
    switch(mode) {
    case 0:return spin::SpinMomentProblem::staticBalance(spin::StaticObjective::MaximumMass,scale);
    case 1:return spin::SpinMomentProblem::staticBalance(spin::StaticObjective::LowerCenterOfMass,scale);
    case 2:return spin::SpinMomentProblem::staticBalancePareto(weight,scale);
    case 3:return spin::SpinMomentProblem::fidgetSpinner();
    case 4:return spin::SpinMomentProblem::yoYo();
    case 5:return spin::SpinMomentProblem::spinningTop(weight);
    case 6:return spin::SpinMomentProblem::spinningTop(weight,spin::SpinningTopObjective::LowerCenterOfMass);
    default:throw std::invalid_argument("mode");
    }
}
template<class T>void emit(const T& x) {
    for(Eigen::Index row=0;row<x.rows();++row)
        for(Eigen::Index col=0;col<x.cols();++col)std::cout<<x(row,col)<<' ';
}
int main(int argc,char**argv) {
    try {
        if(argc!=4)return 2;
        spin::SpinMomentProblem::Vector10 m;
        for(int i=0;i<10;++i)if(!(std::cin>>m(i)))return 3;
        const auto value=select(std::stoi(argv[1]),std::stod(argv[2]),std::stod(argv[3])).evaluate(m);
        std::cout<<std::setprecision(17)<<value.constraints.size()<<' '<<value.objective<<' ';
        emit(value.objectiveGradient);emit(value.objectiveHessian);
        emit(value.constraints);emit(value.constraintJacobian);
        for(const auto& h:value.constraintHessians)emit(h);
        std::cout<<'\n';return 0;
    }catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}
}
'''


@pytest.fixture(scope="module")
def native_objective_oracle(tmp_path_factory):
    eigen_value = os.environ.get("CPINO_EIGEN_ROOT")
    if not eigen_value:
        pytest.skip("CPINO_EIGEN_ROOT not supplied; no independent compiled-native claim")
    eigen = Path(eigen_value).resolve()
    assert (eigen/"Eigen/Core").is_file(), "explicit CPINO_EIGEN_ROOT must contain Eigen/Core"
    compiler = shutil.which("g++")
    assert compiler, "g++ required for explicitly configured native oracle"
    root = Path(__file__).resolve().parents[3]
    directory = tmp_path_factory.mktemp("residual-native-oracle")
    source = directory/"oracle.cpp"; source.write_text(_CPP_ORACLE)
    binary = directory/"oracle"
    command = [compiler,"-std=c++17","-O0","-I"+str(eigen),"-I"+str(root/"include"),
               str(source),str(root/"src/spin/SpinMomentProblem.cpp"),"-o",str(binary)]
    result = subprocess.run(command,capture_output=True,text=True,timeout=60)
    assert result.returncode == 0,result.stdout+result.stderr
    def evaluate(moments,objective):
        output = subprocess.run([str(binary),str(MODES.index(objective["mode"])),
                                 str(objective["weight"]),str(objective["mass_scale"])],
                                input=" ".join(map(str,moments)),capture_output=True,text=True,timeout=10)
        assert output.returncode == 0,output.stderr
        values = np.fromstring(output.stdout,sep=" ")
        count = int(values[0]); cursor = 2
        def consume(shape):
            nonlocal cursor
            size = int(np.prod(shape)); answer = values[cursor:cursor+size].reshape(shape)
            cursor += size
            return answer
        answer = dict(objective=values[1],gradient=consume((10,)),hessian=consume((10,10)),
                      constraints=consume((count,)),dc=consume((count,10)),
                      constraint_hessians=consume((count,10,10)))
        assert cursor == len(values)
        return answer
    return evaluate


@pytest.mark.parametrize("mode",MODES)
def test_compiled_unmodified_native_values_and_loss_hessian_chain(native_objective_oracle,record_property,mode):
    target = exact_target(mode)
    prediction = torch.tensor(NATIVE+DELTA,dtype=torch.float64,requires_grad=True)
    terms = residual_match_terms(prediction,NATIVE,target,coefficients=COEFFICIENTS)
    native = native_objective_oracle(NATIVE,target["objective"])
    predicted = native_objective_oracle(NATIVE+DELTA,target["objective"])
    lam = np.asarray(target["multipliers"])
    r_native = native["gradient"]-native["dc"].T@lam+target["multiplier_mu"]*DUAL
    r_predicted = predicted["gradient"]-predicted["dc"].T@lam+target["multiplier_mu"]*DUAL
    np.testing.assert_allclose(terms["reference_stationarity"].detach(),r_native,rtol=2e-13,atol=2e-14)
    np.testing.assert_allclose(terms["predicted_stationarity"].detach(),r_predicted,rtol=2e-13,atol=2e-14)
    np.testing.assert_allclose(terms["predicted_constraints"].detach(),predicted["constraints"],rtol=2e-13,atol=2e-14)
    dr = (predicted["gradient"]-predicted["dc"].T@lam)-(native["gradient"]-native["dc"].T@lam)
    dc = predicted["constraints"]-native["constraints"]
    hessian_l = predicted["hessian"]-np.einsum("i,ijk->jk",lam,predicted["constraint_hessians"])
    expected_gradient = (2*hessian_l.T@(dr/np.asarray(target["stationarity_scale"])**2)
                         +2*predicted["dc"].T@(dc/np.asarray(target["constraint_scale"])**2))
    loss = terms["stationarity_scaled_delta"].square().sum()+terms["constraint_scaled_delta"].square().sum()
    loss.backward()
    np.testing.assert_allclose(prediction.grad,expected_gradient,rtol=2e-10,atol=3e-13)
    record_property("native_profile",REPO_PROFILE)
    record_property("stationarity_max_absolute_error",float(np.max(np.abs(terms["predicted_stationarity"].detach().numpy()-r_predicted))))
    record_property("constraint_max_absolute_error",float(np.max(np.abs(terms["predicted_constraints"].detach().numpy()-predicted["constraints"]))))
    record_property("loss_gradient_max_absolute_error",float(np.max(np.abs(prediction.grad.numpy()-expected_gradient))))
