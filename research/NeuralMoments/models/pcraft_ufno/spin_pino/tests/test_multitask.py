import pytest
import torch

from research.NeuralMoments.models.pcraft_ufno.spin_pino.multitask import (
    fixed_weighted_objective, task_gradient_matrix, pcgrad_project,
    GradNormBalancer, LearnedTaskWeights, event_positive_class_weight,
)


def test_fixed_weighted_objective_matches_manual_sum_and_validates_weights():
    a=torch.tensor(2.,requires_grad=True);b=torch.tensor(3.,requires_grad=True)
    out=fixed_weighted_objective({"moment":a,"event":b},{"moment":1.,"event":.2})
    assert out.item()==pytest.approx(2.6)
    with pytest.raises(ValueError,match="invalid task weight"):
        fixed_weighted_objective({"moment":a},{"moment":-1.})


def test_pcgrad_projects_conflicting_gradients_and_reports_cosines():
    g1=torch.tensor([1.,0.]);g2=torch.tensor([-1.,1.])
    combined,report=pcgrad_project([g1,g2])
    assert report["cosine_matrix"][0][1]<0
    assert report["projection_counts"][0]==1
    assert torch.isfinite(combined).all()
    assert combined[0]>0


def test_pcgrad_is_identity_for_nonconflicting_gradients():
    g1=torch.tensor([1.,0.]);g2=torch.tensor([0.,2.])
    combined,report=pcgrad_project([g1,g2])
    assert torch.allclose(combined,(g1+g2)/2)
    assert report["projection_counts"]==[0,0]


def test_task_gradient_matrix_keeps_task_gradients_separate():
    p=torch.nn.Parameter(torch.tensor([1.,2.]))
    l1=(p[0]-3).square();l2=(p[1]+1).square()
    names,rows=task_gradient_matrix({"moment":l1,"jacobian":l2},[p])
    assert names==["moment","jacobian"]
    assert torch.equal(rows[0],torch.tensor([-4.,0.]))
    assert torch.equal(rows[1],torch.tensor([0.,6.]))


def test_gradnorm_balancer_ignores_absent_tasks_and_normalizes_active_weights():
    balancer=GradNormBalancer(("moment","jacobian","event"))
    before=dict(balancer.weights)
    after=balancer.update({"moment":2.,"jacobian":1.},{"moment":4.,"jacobian":2.})
    assert after["event"]==before["event"]
    assert sum(after[k] for k in ("moment","jacobian"))==pytest.approx(2.)
    assert all(v>0 for v in after.values())


def test_learned_task_weights_are_distinct_from_predictive_uq_and_have_gradients():
    module=LearnedTaskWeights(("moment","jacobian"))
    losses={"moment":torch.tensor(2.,requires_grad=True),
            "jacobian":torch.tensor(3.,requires_grad=True)}
    total=module(losses)
    total.backward()
    assert torch.isfinite(total)
    assert all(p.grad is not None and torch.isfinite(p.grad) for p in module.parameters())


def test_missing_task_is_excluded_and_empty_task_set_rejected():
    module=LearnedTaskWeights(("moment","event"))
    out=module({"moment":torch.tensor(1.,requires_grad=True),"event":None})
    assert torch.isfinite(out)
    with pytest.raises(ValueError,match="at least one"):
        module({"moment":None,"event":None})


def test_gradnorm_rejects_nonfinite_statistics():
    balancer=GradNormBalancer(("moment",))
    with pytest.raises(ValueError,match="invalid GradNorm"):
        balancer.update({"moment":float("nan")},{"moment":1.})


def test_pcgrad_assignment_projects_shared_gradients_and_retains_private_head_gradients():
    from research.NeuralMoments.models.pcraft_ufno.spin_pino.multitask import pcgrad_assign_gradients
    shared=torch.nn.Parameter(torch.tensor([1.,0.]))
    event_head=torch.nn.Parameter(torch.tensor(2.))
    uq_head=torch.nn.Parameter(torch.tensor(3.))
    moment=(shared[0]+0*event_head).sum()
    event=(-shared[0]+2*event_head)
    uq=(shared[1]+3*uq_head)
    report=pcgrad_assign_gradients(
        {"moment":moment,"event":event,"uq":uq},
        {"moment":1.,"event":1.,"uq":.5},
        [shared],[event_head,uq_head])
    assert report["task_names"]==["moment","event","uq"]
    assert report["projection_counts"][0]>=1
    assert shared.grad is not None and torch.isfinite(shared.grad).all()
    assert event_head.grad.item()==pytest.approx(2.)
    assert uq_head.grad.item()==pytest.approx(1.5)


def test_pcgrad_assignment_rejects_overlapping_or_duplicate_parameter_groups():
    from research.NeuralMoments.models.pcraft_ufno.spin_pino.multitask import pcgrad_assign_gradients
    p=torch.nn.Parameter(torch.ones(1))
    loss=p.sum()
    with pytest.raises(ValueError,match="overlap"):
        pcgrad_assign_gradients({"x":loss},{"x":1.},[p],[p])
    with pytest.raises(ValueError,match="duplicate"):
        pcgrad_assign_gradients({"x":loss},{"x":1.},[p,p],[])


def test_pcgrad_assignment_skips_absent_tasks_and_updates_optimizer():
    from research.NeuralMoments.models.pcraft_ufno.spin_pino.multitask import pcgrad_assign_gradients
    shared=torch.nn.Parameter(torch.tensor([1.,2.]))
    head=torch.nn.Parameter(torch.tensor(1.))
    optimizer=torch.optim.SGD([shared,head],lr=.1)
    optimizer.zero_grad(set_to_none=True)
    active=(shared.square().sum()+head.square())
    report=pcgrad_assign_gradients({"moment":active,"event":None},
        {"moment":1.,"event":1.},[shared],[head])
    before=shared.detach().clone()
    optimizer.step()
    assert report["task_names"]==["moment"]
    assert not torch.equal(before,shared.detach())
    assert head.grad is not None


def test_pcgrad_assignment_rejects_empty_active_task_set():
    from research.NeuralMoments.models.pcraft_ufno.spin_pino.multitask import pcgrad_assign_gradients
    p=torch.nn.Parameter(torch.ones(1))
    with pytest.raises(ValueError,match="no active"):
        pcgrad_assign_gradients({"event":None},{"event":1.},[p],[])


def test_native_task_loss_decomposition_reconstructs_fixed_objective():
    import numpy as np
    from research.NeuralMoments.models.pcraft_ufno.spin_pino.native_train import loss_for_batch
    from research.NeuralMoments.models.pcraft_ufno.spin_pino.native_pcraft import (
        NativePCRAFTConfig,PcraftNativeMomentOperator)
    from research.NeuralMoments.models.pcraft_ufno.spin_pino.tests.test_native_pcraft import small_geometry
    cfg=NativePCRAFTConfig(grid_size=8,pino_width=2,pino_modes=1,pino_layers=2,
        interval_width=8,interval_depth=1,chunk_intervals=16,min_native_resolution=1,
        event_validity_head_enabled=True)
    model=PcraftNativeMomentOperator(cfg).train()
    cache=model.encode(small_geometry(),chunk=16)
    q=np.asarray([[0.,1.,0,0,0,0,0,0,0,0],[0.,0,1.,0,0,0,0,0,0,0]],dtype=np.float64)
    labels={"queries":q,"moments":np.zeros((2,10)),"jacobian":np.ones((2,10,10)),
        "derivative_valid":np.asarray([True,False]),"event_valid":np.asarray([True,False]),
        "event_label_valid":np.asarray([True,True])}
    fixed,*_=loss_for_batch(model,cache,labels,np.arange(2),.05,"cpu",event_weight=.1)
    tasks,counts=loss_for_batch(model,cache,labels,np.arange(2),.05,"cpu",
        event_weight=.1,return_task_losses=True)
    reconstructed=tasks["moment"]+.05*tasks["jacobian"]+.1*tasks["event"]
    assert torch.allclose(fixed,reconstructed,rtol=1e-12,atol=1e-12)
    assert counts["jacobian"]==1 and counts["event"]==2


def test_gradnorm_state_roundtrip_restores_weights_and_initial_losses():
    balancer=GradNormBalancer(("moment","event"))
    balancer.update({"moment":2.,"event":1.},{"moment":3.,"event":2.})
    state=balancer.state_dict()
    restored=GradNormBalancer(("moment","event"))
    restored.load_state_dict(state)
    assert restored.state_dict()==state
    with pytest.raises(ValueError,match="task-name contract"):
        GradNormBalancer(("moment",)).load_state_dict(state)


def test_gradnorm_state_rejects_hyperparameter_mismatch_and_nonfinite_weights():
    balancer=GradNormBalancer(("moment",))
    state=balancer.state_dict()
    state["weights"]["moment"]=float("nan")
    with pytest.raises(ValueError,match="invalid weights"):
        balancer.load_state_dict(state)
    state=balancer.state_dict()
    state["alpha"]=2.0
    with pytest.raises(ValueError,match="hyperparameter contract"):
        balancer.load_state_dict(state)


def test_event_positive_weight_uses_only_labeled_training_rows_and_is_bounded():
    report=event_positive_class_weight(
        [True,False,False,False,True,True],
        [True,True,True,False,True,True],min_weight=.5,max_weight=2.)
    assert report["positive_count"]==3
    assert report["negative_count"]==2
    assert report["unknown_count"]==1
    assert report["positive_weight"]==pytest.approx(2/3)
    assert report["source"]=="labeled_training_rows_only"


def test_event_positive_weight_clips_extreme_imbalance_and_rejects_missing_class():
    report=event_positive_class_weight([True,False,False,False,False],
        min_weight=.25,max_weight=2.)
    assert report["positive_weight"]==2.
    assert report["clipped"] is True
    with pytest.raises(ValueError,match="without positive"):
        event_positive_class_weight([False,False])
    with pytest.raises(ValueError,match="contain labeled"):
        event_positive_class_weight([True,False],[False,False])


def test_learned_task_weight_state_restores_separately_from_predictive_uq():
    module=LearnedTaskWeights(("moment","jacobian"))
    optimizer=torch.optim.AdamW(module.parameters(),lr=.01)
    losses={"moment":torch.tensor(2.,requires_grad=True),
            "jacobian":torch.tensor(1.,requires_grad=True)}
    optimizer.zero_grad(set_to_none=True)
    module(losses).backward()
    optimizer.step()
    state=module.state_dict()
    restored=LearnedTaskWeights(("moment","jacobian"))
    restored.load_state_dict(state)
    for key,value in state.items():
        assert torch.equal(restored.state_dict()[key],value)
