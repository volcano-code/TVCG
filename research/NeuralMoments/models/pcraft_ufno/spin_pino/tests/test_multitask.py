import pytest
import torch

from research.NeuralMoments.models.pcraft_ufno.spin_pino.multitask import (
    fixed_weighted_objective, task_gradient_matrix, pcgrad_project,
    GradNormBalancer, LearnedTaskWeights,
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
