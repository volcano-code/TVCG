"""Resume and parameter ownership contracts for unified joint training."""
from __future__ import annotations

from copy import deepcopy
from dataclasses import replace
import json
import random

import numpy as np
import pytest
import torch

from research.NeuralMoments.models.pcraft_ufno.spin_pino.native_pcraft import PcraftNativeMomentOperator
from research.NeuralMoments.models.pcraft_ufno.candidate_projects.quadrann_reproduction.spin_adapter.tests.test_unified_m80_sqno import (
    make_geometry, make_unified, queries,
)


DATA_FINGERPRINT="unified-training-state-synthetic-v1"


def _batch():
    q=queries()[:2]
    return (q,torch.zeros(2,10,dtype=torch.float64),
            torch.zeros(2,10,10,dtype=torch.float64))


def _optimizer_and_scheduler(model,*,reverse=False):
    if reverse:
        optimizer=torch.optim.AdamW(list(reversed(list(model.unique_trainable_parameters()))),
                                    lr=1e-4,weight_decay=.01)
    else:
        optimizer=model.build_optimizer(learning_rate=1e-4,weight_decay=.01)
    return optimizer,torch.optim.lr_scheduler.StepLR(optimizer,step_size=1,gamma=.7)


def _step(model,optimizer,batch,*,gradient_mode="pcgrad",**settings):
    return model.train_step(optimizer,model.encode_geometry(make_geometry()),*batch,
                            gradient_mode=gradient_mode,**settings)


def _state(model,optimizer,scheduler):
    return deepcopy({"model":model.state_dict(),"optimizer":optimizer.state_dict(),
                     "scheduler":scheduler.state_dict(),"torch_rng":torch.get_rng_state(),
                     "numpy_rng":np.random.get_state(),"python_rng":random.getstate()})


def _assert_identical(actual,expected):
    if isinstance(expected,torch.Tensor):
        torch.testing.assert_close(actual,expected,rtol=0,atol=0)
    elif isinstance(expected,np.ndarray):
        np.testing.assert_array_equal(actual,expected)
    elif isinstance(expected,dict):
        assert actual.keys()==expected.keys()
        for key in expected:
            _assert_identical(actual[key],expected[key])
    elif isinstance(expected,(tuple,list)):
        assert type(actual) is type(expected) and len(actual)==len(expected)
        for a,b in zip(actual,expected):
            _assert_identical(a,b)
    else:
        assert actual==expected


def _trained_checkpoint(tmp_path,*,gradient_mode="pcgrad"):
    model=make_unified().train()
    optimizer,scheduler=_optimizer_and_scheduler(model)
    batch=_batch()
    _step(model,optimizer,batch,gradient_mode=gradient_mode)
    scheduler.step()
    path=tmp_path/"trained.pt"
    model.save_checkpoint(path,optimizer=optimizer,scheduler=scheduler,
                          data_fingerprint=DATA_FINGERPRINT)
    return path,model,optimizer,scheduler,batch


@pytest.mark.parametrize("gradient_mode",["fixed","pcgrad"])
def test_unified_training_resume_reproduces_next_update_optimizer_scheduler_and_rng(tmp_path,gradient_mode):
    path,model,optimizer,scheduler,batch=_trained_checkpoint(tmp_path,gradient_mode=gradient_mode)
    contract=model.training_contract(gradient_mode=gradient_mode)
    # Seed all three streams after fixture construction, then save their state.
    torch.manual_seed(197)
    np.random.seed(198)
    random.seed(199)
    model.save_checkpoint(path,optimizer=optimizer,scheduler=scheduler,
                          data_fingerprint=DATA_FINGERPRINT)
    saved=_state(model,optimizer,scheduler)
    assert optimizer.state
    assert any(p.is_complex() and p in optimizer.state for p in model.parameters())
    expected_draws=(torch.rand(5),np.random.random(5),random.random())
    expected_metrics=_step(model,optimizer,batch,gradient_mode=gradient_mode)
    scheduler.step()
    expected=_state(model,optimizer,scheduler)

    restored=make_unified().train()
    restored_opt,restored_scheduler=_optimizer_and_scheduler(restored)
    restored.load_checkpoint(path,optimizer=restored_opt,scheduler=restored_scheduler,
        expected_data_fingerprint=DATA_FINGERPRINT,expected_training_contract=contract,restore_rng=True)
    _assert_identical(_state(restored,restored_opt,restored_scheduler),saved)
    actual_draws=(torch.rand(5),np.random.random(5),random.random())
    _assert_identical(actual_draws,expected_draws)
    actual_metrics=_step(restored,restored_opt,batch,gradient_mode=gradient_mode)
    restored_scheduler.step()
    _assert_identical(_state(restored,restored_opt,restored_scheduler),expected)
    assert actual_metrics["loss"]==expected_metrics["loss"]
    assert actual_metrics["gradient_norm"]==expected_metrics["gradient_norm"]


@pytest.mark.parametrize("change",[
    {"gradient_mode":"fixed"},{"jacobian_weight":.2},{"branch_aux_weight":.3},
    {"pcgrad_reduction":"mean"},{"gradient_clip":2.},
    {"moment_scale":torch.linspace(1.,2.,10)},
    {"jacobian_scale":torch.full((10,10),2.)},
],ids=["mode","jacobian_weight","branch_weight","reduction","clip","moment_scale","jacobian_scale"])
def test_unified_resume_rejects_training_contract_drift_before_any_state_mutation(tmp_path,change):
    path,_,_,_,_=_trained_checkpoint(tmp_path)
    restored=make_unified().train()
    with torch.no_grad():
        restored.fusion_logit.add_(.37)
    optimizer,scheduler=_optimizer_and_scheduler(restored)
    expected=restored.training_contract(**({"gradient_mode":"pcgrad"}|change))
    before=_state(restored,optimizer,scheduler)
    with pytest.raises(ValueError,match="contract"):
        restored.load_checkpoint(path,optimizer=optimizer,scheduler=scheduler,
            expected_data_fingerprint=DATA_FINGERPRINT,expected_training_contract=expected,restore_rng=True)
    _assert_identical(_state(restored,optimizer,scheduler),before)


def test_unified_resume_rejects_reordered_optimizer_parameters_before_mutation(tmp_path):
    path,_,_,_,_=_trained_checkpoint(tmp_path)
    restored=make_unified().train()
    optimizer,scheduler=_optimizer_and_scheduler(restored,reverse=True)
    before=_state(restored,optimizer,scheduler)
    with pytest.raises(ValueError,match="(?i)optimizer"):
        restored.load_checkpoint(path,optimizer=optimizer,scheduler=scheduler,
            expected_data_fingerprint=DATA_FINGERPRINT,
            expected_training_contract=restored.training_contract(gradient_mode="pcgrad"),restore_rng=True)
    _assert_identical(_state(restored,optimizer,scheduler),before)


@pytest.mark.parametrize("missing",["optimizer","scheduler"])
def test_unified_resume_rejects_missing_component_before_mutation(tmp_path,missing):
    path,_,_,_,_=_trained_checkpoint(tmp_path)
    payload=torch.load(path,weights_only=False)
    payload[missing]=None
    torch.save(payload,path)
    restored=make_unified().train()
    with torch.no_grad():
        restored.fusion_logit.add_(.37)
    optimizer,scheduler=_optimizer_and_scheduler(restored)
    before=_state(restored,optimizer,scheduler)
    with pytest.raises(ValueError,match=missing):
        restored.load_checkpoint(path,optimizer=optimizer,scheduler=scheduler,
            expected_data_fingerprint=DATA_FINGERPRINT,
            expected_training_contract=restored.training_contract(gradient_mode="pcgrad"),restore_rng=True)
    _assert_identical(_state(restored,optimizer,scheduler),before)


def test_unified_resume_requires_explicit_expected_training_contract(tmp_path):
    path,_,_,_,_=_trained_checkpoint(tmp_path)
    restored=make_unified().train()
    optimizer,scheduler=_optimizer_and_scheduler(restored)
    before=_state(restored,optimizer,scheduler)
    with pytest.raises(ValueError,match="contract"):
        restored.load_checkpoint(path,optimizer=optimizer,scheduler=scheduler,
                                expected_data_fingerprint=DATA_FINGERPRINT,restore_rng=True)
    _assert_identical(_state(restored,optimizer,scheduler),before)


@pytest.mark.parametrize("fingerprint",[None,""])
def test_unified_strict_training_save_requires_data_fingerprint(tmp_path,fingerprint):
    _,model,optimizer,scheduler,_=_trained_checkpoint(tmp_path)
    path=tmp_path/"missing-data.pt"
    with pytest.raises(ValueError,match="(?i)data.*fingerprint"):
        model.save_checkpoint(path,optimizer=optimizer,scheduler=scheduler,data_fingerprint=fingerprint)
    assert not path.exists()


@pytest.mark.parametrize("fingerprint",[None,""])
def test_unified_strict_resume_requires_expected_data_fingerprint_before_mutation(tmp_path,fingerprint):
    path,_,_,_,_=_trained_checkpoint(tmp_path)
    restored=make_unified().train()
    optimizer,scheduler=_optimizer_and_scheduler(restored)
    before=_state(restored,optimizer,scheduler)
    with pytest.raises(ValueError,match="(?i)data.*fingerprint"):
        restored.load_checkpoint(path,optimizer=optimizer,scheduler=scheduler,
            expected_data_fingerprint=fingerprint,
            expected_training_contract=restored.training_contract(gradient_mode="pcgrad"),restore_rng=True)
    _assert_identical(_state(restored,optimizer,scheduler),before)


def test_unified_resumed_step_rejects_mode_drift_without_consuming_cache(tmp_path):
    path,_,_,_,batch=_trained_checkpoint(tmp_path)
    restored=make_unified().train()
    optimizer,scheduler=_optimizer_and_scheduler(restored)
    restored.load_checkpoint(path,optimizer=optimizer,scheduler=scheduler,
        expected_data_fingerprint=DATA_FINGERPRINT,
        expected_training_contract=restored.training_contract(gradient_mode="pcgrad"),restore_rng=True)
    cache=restored.encode_geometry(make_geometry())
    before=_state(restored,optimizer,scheduler)
    with pytest.raises(ValueError,match="contract"):
        restored.train_step(optimizer,cache,*batch,gradient_mode="fixed")
    _assert_identical(_state(restored,optimizer,scheduler),before)
    metrics=restored.train_step(optimizer,cache,*batch,gradient_mode="pcgrad")
    assert metrics["gradient_mode"]=="pcgrad"
    assert metrics["multitask_report"]["reduction"]=="sum"


def test_unified_explicit_training_contract_roundtrip_and_inference_load(tmp_path):
    model=make_unified().train()
    optimizer,scheduler=_optimizer_and_scheduler(model)
    contract=model.training_contract(gradient_mode="pcgrad")
    json.dumps(contract,allow_nan=False)
    path=tmp_path/"explicit.pt"
    model.save_checkpoint(path,optimizer=optimizer,scheduler=scheduler,
                          data_fingerprint=DATA_FINGERPRINT,training_contract=contract)
    restored=make_unified().train()
    restored_opt,restored_scheduler=_optimizer_and_scheduler(restored)
    restored.load_checkpoint(path,optimizer=restored_opt,scheduler=restored_scheduler,
        expected_data_fingerprint=DATA_FINGERPRINT,expected_training_contract=contract)
    _assert_identical(restored.state_dict(),model.state_dict())
    inference=make_unified().eval()
    inference.load_checkpoint(path,expected_data_fingerprint=DATA_FINGERPRINT)
    _assert_identical(inference.state_dict(),model.state_dict())


def test_unified_save_rejects_contract_different_from_last_successful_step(tmp_path):
    _,model,optimizer,scheduler,_=_trained_checkpoint(tmp_path)
    path=tmp_path/"wrong-mode.pt"
    with pytest.raises(ValueError,match="contract"):
        model.save_checkpoint(path,optimizer=optimizer,scheduler=scheduler,
            data_fingerprint=DATA_FINGERPRINT,
            training_contract=model.training_contract(gradient_mode="fixed"))
    assert not path.exists()


@pytest.mark.parametrize("gradient_mode",["fixed","pcgrad"])
def test_unified_training_partition_is_complete_and_preserves_inactive_native_heads(gradient_mode):
    model=make_unified().train()
    model.shared_encoder=PcraftNativeMomentOperator(replace(model.shared_encoder.config,
        event_validity_head_enabled=True,uq_enabled=True))
    partition=model.training_parameter_partition()
    all_names=[name for name,p in model.named_parameters() if p.requires_grad]
    combined=partition["shared"]+partition["private"]+partition["inactive"]
    assert set(combined)==set(all_names)
    assert len(combined)==len(set(combined))==len(all_names)
    assert partition["private"]==[]
    expected_inactive=[name for name in all_names if name.startswith(
        ("shared_encoder.event_validity_head.","shared_encoder.uq_head."))]
    assert partition["inactive"]==expected_inactive
    assert "fusion_logit" in partition["shared"]
    assert any(name.startswith("shared_encoder.convex_potential_head.") for name in partition["shared"])
    assert any(name.startswith("sqno.") for name in partition["shared"])
    params=dict(model.named_parameters())
    before={name:params[name].detach().clone() for name in expected_inactive}
    optimizer=model.build_optimizer(learning_rate=1e-4,weight_decay=.1)
    _step(model,optimizer,_batch(),gradient_mode=gradient_mode)
    for name,value in before.items():
        assert params[name].grad is None
        assert torch.equal(params[name],value)
        assert params[name] not in optimizer.state


@pytest.mark.parametrize("fusion_mode",["global_scalar","query_conditioned"])
def test_unified_joint_loss_reports_actual_fusion_and_independent_valid_counts(fusion_mode):
    model=make_unified(fusion_mode=fusion_mode).train()
    cache=model.encode_geometry(make_geometry())
    q=queries()
    result=model.joint_training_loss(cache,q,torch.zeros(4,10,dtype=torch.float64),
        torch.zeros(4,10,10,dtype=torch.float64),
        valid_mask=torch.tensor([True,False,True,True]),
        jacobian_valid_mask=torch.tensor([False,True,False,True]))
    assert result["valid_queries"].item()==3
    assert result["valid_jacobian_queries"].item()==2
    expected=model.evaluate_queries(cache,q,create_graph=True)["fusion_weight"].detach()
    torch.testing.assert_close(result["fusion_weight"],expected)
    assert torch.all(result["fusion_weight"]>0)
    assert all(isinstance(value,torch.Tensor) and torch.isfinite(value).all()
               for value in result.values())


@pytest.mark.parametrize("tamper",[
    "optimizer_lr","optimizer_parameter_order","missing_exp_avg",
    "missing_exp_avg_sq","missing_step","scheduler_gamma",
])
def test_unified_resume_rejects_raw_state_disagreement_with_contract_before_mutation(tmp_path,tamper):
    path,_,_,_,_=_trained_checkpoint(tmp_path)
    payload=torch.load(path,weights_only=False)
    if tamper=="optimizer_lr":
        payload["optimizer"]["param_groups"][0]["lr"]*=2
    elif tamper=="optimizer_parameter_order":
        parameter_ids=payload["optimizer"]["param_groups"][0]["params"]
        # Swap equal-shaped states so a tensor-shape check alone cannot detect
        # assigning one parameter's AdamW moments to another parameter.
        first_by_shape={}
        pair=None
        for index,parameter_id in enumerate(parameter_ids):
            state=payload["optimizer"]["state"].get(parameter_id,{})
            if "exp_avg" not in state:
                continue
            value=state["exp_avg"]
            signature=(tuple(value.shape),value.dtype)
            if signature in first_by_shape:
                pair=(first_by_shape[signature],index)
                break
            first_by_shape[signature]=index
        assert pair is not None
        i,j=pair
        parameter_ids[i],parameter_ids[j]=parameter_ids[j],parameter_ids[i]
    elif tamper.startswith("missing_"):
        state=next(iter(payload["optimizer"]["state"].values()))
        del state[tamper.removeprefix("missing_")]
    else:
        payload["scheduler"]["gamma"]*=.5
    # Keep all metadata intact: validation must compare it with the raw states
    # that will actually be handed to optimizer.load_state_dict and the scheduler.
    torch.save(payload,path)
    restored=make_unified().train()
    with torch.no_grad():
        restored.fusion_logit.add_(.37)
    optimizer,scheduler=_optimizer_and_scheduler(restored)
    before=_state(restored,optimizer,scheduler)
    with pytest.raises(ValueError,match="(?i)optimizer|scheduler|state"):
        restored.load_checkpoint(path,optimizer=optimizer,scheduler=scheduler,
            expected_data_fingerprint=DATA_FINGERPRINT,
            expected_training_contract=restored.training_contract(gradient_mode="pcgrad"),restore_rng=True)
    _assert_identical(_state(restored,optimizer,scheduler),before)


@pytest.mark.parametrize("gradient_mode",["fixed","pcgrad"])
def test_unified_trained_model_preserves_scale_invariance_and_raw_jacobian(gradient_mode):
    model=make_unified().train()
    optimizer=model.build_optimizer(learning_rate=1e-4)
    batch=_batch()
    _step(model,optimizer,batch,gradient_mode=gradient_mode)
    model.eval()
    with torch.no_grad():
        cache=model.encode_geometry(make_geometry(),inference=True)
        q=batch[0]
        base=model.evaluate_queries(cache,q)
        scale=3.7
        scaled=model.evaluate_queries(cache,q*scale)
        torch.testing.assert_close(base["moments"],scaled["moments"],rtol=2e-7,atol=2e-9)
        torch.testing.assert_close(base["jacobian_coefficients"]/scale,
                                   scaled["jacobian_coefficients"],rtol=2e-5,atol=2e-7)
        eps=2e-6
        for column in (0,4,8):
            delta=torch.zeros_like(q[:1])
            delta[0,column]=eps
            plus=model.evaluate_queries(cache,q[:1]+delta)
            minus=model.evaluate_queries(cache,q[:1]-delta)
            finite_difference=(plus["moments"]-minus["moments"])/(2*eps)
            torch.testing.assert_close(base["jacobian_coefficients"][:1,:,column],
                                       finite_difference,rtol=4e-3,atol=3e-5)


@pytest.mark.parametrize("kind",["lambda","multiplicative","sequential","chained"])
def test_unified_strict_checkpoint_rejects_unverifiable_callable_schedulers(tmp_path,kind):
    model=make_unified().train()
    optimizer=model.build_optimizer(learning_rate=1e-4)
    if kind=="lambda":
        scheduler=torch.optim.lr_scheduler.LambdaLR(optimizer,lambda epoch: .8**epoch)
    elif kind=="multiplicative":
        scheduler=torch.optim.lr_scheduler.MultiplicativeLR(optimizer,lambda epoch: .8)
    elif kind=="sequential":
        scheduler=torch.optim.lr_scheduler.SequentialLR(optimizer,[
            torch.optim.lr_scheduler.ConstantLR(optimizer,factor=.5,total_iters=2),
            torch.optim.lr_scheduler.LambdaLR(optimizer,lambda epoch: .8**epoch),
        ],milestones=[2])
    else:
        scheduler=torch.optim.lr_scheduler.ChainedScheduler([
            torch.optim.lr_scheduler.StepLR(optimizer,step_size=2,gamma=.7),
            torch.optim.lr_scheduler.MultiplicativeLR(optimizer,lambda epoch: .8),
        ])
    before=_state(model,optimizer,scheduler)
    path=tmp_path/"callable-scheduler.pt"
    with pytest.raises(ValueError,match="(?i)callable|lambda|scheduler"):
        model.save_checkpoint(path,optimizer=optimizer,scheduler=scheduler,
            data_fingerprint=DATA_FINGERPRINT,training_contract=model.training_contract())
    assert not path.exists()
    _assert_identical(_state(model,optimizer,scheduler),before)


@pytest.mark.parametrize("kind",["sequential","chained"])
def test_unified_composed_state_driven_scheduler_resume_matches_next_updates(tmp_path,kind):
    def setup(model):
        optimizer=model.build_optimizer(learning_rate=1e-4,weight_decay=.01)
        children=[torch.optim.lr_scheduler.StepLR(optimizer,step_size=2,gamma=.7),
                  torch.optim.lr_scheduler.ExponentialLR(optimizer,gamma=.9)]
        scheduler=(torch.optim.lr_scheduler.SequentialLR(optimizer,children,milestones=[2])
                   if kind=="sequential" else torch.optim.lr_scheduler.ChainedScheduler(children))
        return optimizer,scheduler

    model=make_unified().train()
    optimizer,scheduler=setup(model)
    batch=_batch()
    _step(model,optimizer,batch,gradient_mode="fixed")
    scheduler.step()
    path=tmp_path/"composed.pt"
    contract=model.training_contract(gradient_mode="fixed")
    model.save_checkpoint(path,optimizer=optimizer,scheduler=scheduler,
                          data_fingerprint=DATA_FINGERPRINT)
    expected=[]
    # Cross SequentialLR's child-switch boundary, then use the second schedule.
    for _ in range(2):
        _step(model,optimizer,batch,gradient_mode="fixed")
        scheduler.step()
        expected.append(_state(model,optimizer,scheduler))
    restored=make_unified().train()
    restored_opt,restored_scheduler=setup(restored)
    restored.load_checkpoint(path,optimizer=restored_opt,scheduler=restored_scheduler,
        expected_data_fingerprint=DATA_FINGERPRINT,expected_training_contract=contract,restore_rng=True)
    for state in expected:
        _step(restored,restored_opt,batch,gradient_mode="fixed")
        restored_scheduler.step()
        _assert_identical(_state(restored,restored_opt,restored_scheduler),state)
