from __future__ import annotations

import copy
from dataclasses import replace
import json
import math
import weakref

import numpy as np
import pytest
import torch

from experiments.compact_pino_v21.training.losses import physical_loss
from experiments.compact_pino_v21.training.state import capture_rng
from experiments.henmo_v1plus.diagnostics import (
    gradient_alignment, run_gradient_diagnostics, run_path_diagnostics,
)
from experiments.henmo_v1plus.fixtures import synthetic_dataset
from research.NeuralMoments.models.pcraft_ufno.candidate_projects.quadrann_reproduction.spin_adapter.henmo_config import HENMOConfig
from research.NeuralMoments.models.pcraft_ufno.candidate_projects.quadrann_reproduction.spin_adapter.henmo_operator import HENMOOperator


@pytest.fixture
def model():
    old_threads = torch.get_num_threads()
    torch.set_num_threads(2)
    torch.manual_seed(418)
    with HENMOOperator(HENMOConfig.cpu_smoke(response_zero_init=False)).double() as value:
        yield value
    torch.set_num_threads(old_threads)


def _assert_tree_equal(left, right):
    if isinstance(left, torch.Tensor):
        assert torch.equal(left, right)
    elif isinstance(left, dict):
        assert left.keys() == right.keys()
        for key in left:
            _assert_tree_equal(left[key], right[key])
    elif isinstance(left, (tuple, list)):
        assert len(left) == len(right)
        for a, b in zip(left, right):
            _assert_tree_equal(a, b)
    else:
        assert left == right


def _state(model):
    return dict(config=model.config, response=model.response,
                modes=[module.training for module in model.modules()],
                tensors={name: value.detach().clone() for name, value in model.state_dict().items()},
                versions={name: value._version for name, value in model.named_parameters()},
                gradients={name: None if p.grad is None else p.grad.detach().clone()
                           for name, p in model.named_parameters()},
                rng=capture_rng())


def _assert_state_preserved(model, state):
    assert model.config is state['config']
    assert model.response is state['response']
    assert [module.training for module in model.modules()] == state['modes']
    assert {name: value._version for name, value in model.named_parameters()} == state['versions']
    for name, value in model.state_dict().items():
        assert torch.equal(value, state['tensors'][name])
    for name, parameter in model.named_parameters():
        if state['gradients'][name] is None:
            assert parameter.grad is None
        else:
            assert torch.equal(parameter.grad, state['gradients'][name])
    _assert_tree_equal(capture_rng(), state['rng'])
    assert model._feature_entry is None  # A temporary diagnostic graph is never reused.


def test_actual_paired_paths_zero_control_and_state_preservation(model):
    data = synthetic_dataset()
    model.train()
    model.geometry_attention.eval()  # Preserve mixed submodule modes too.
    next(model.parameters()).grad = torch.ones_like(next(model.parameters()))
    state = _state(model)
    report = run_path_diagnostics(model, data.train[:1], coarse_grids=(2, 4), diagnostic_nodes=2)
    assert report['actual_parameter_updates'] == 0
    assert report['physics_promotion'] is False
    assert set(report['routes']) == {
        'coarse2', 'coarse4', 'coarse2_zero_correction', 'coarse4_zero_correction',
        'adaptive2', 'adaptive4', 'deterministic_gl2', 'deterministic_gl3'}
    for name, result in report['routes'].items():
        assert result['complete'] and result['derivative_complete']
        row = result['rows'][0]
        assert row['record_id'] == data.train[0]['record_id']
        assert np.shape(row['predicted_moments']) == (10,)
        assert np.shape(row['predicted_jacobian']) == (10, 3)
        assert row['native_moments'] == data.train[0]['case']['native_moments'].tolist()
        assert row['qualification']['derivative_label_qualified'] is True
        if name.startswith('deterministic'):
            assert row['timing']['geometry_encode_s'] == 0.
            assert row['diagnostics']['exact_hard_mesh_oracle'] is False
        else:
            assert 'stage_profile' in row['timing']
            assert row['timing']['stage_profile']['phases']['geometry_encode']['calls'] == 1
    assert report['routes']['adaptive4']['rows'][0]['adaptive_starts_at_leaf'] is True
    assert report['routes']['adaptive2']['rows'][0]['adaptive_starts_at_leaf'] is False
    for grid in (2, 4):
        learned = report['routes'][f'coarse{grid}']['rows'][0]
        zero = report['routes'][f'coarse{grid}_zero_correction']['rows'][0]
        assert learned['diagnostics']['coarse_grid'] == zero['diagnostics']['coarse_grid'] == grid
        assert learned['diagnostics']['evaluated_atoms'] == zero['diagnostics']['evaluated_atoms']
        assert learned['diagnostics']['coarse_support_fraction'] == zero['diagnostics']['coarse_support_fraction']
        assert not np.array_equal(learned['predicted_moments'], zero['predicted_moments'])
    json.dumps(report, allow_nan=False)
    _assert_state_preserved(model, state)


def test_gradients_match_independent_same_query_autograd_and_preserve_state(model):
    record = synthetic_dataset().train[0]
    case = record['case']
    model.config = replace(model.config, training_execution='coarse', training_coarse_grids=(2,))
    model.train()
    cache = model.encode_geometry(case['blocks'], case['fixed_moments'], density=1., inference=False)
    prediction = model.evaluate_operand(cache, case['operand'], with_jacobian=True,
                                        parameter_scales=case['parameter_scales'], coarse_grid=2)
    m_loss, _ = physical_loss([prediction], [record], jacobian_loss_weight=0.)
    j_loss, _ = physical_loss([prediction], [dict(record, moment_weight=0.)], jacobian_loss_weight=1.)
    parameters = [p for p in model.parameters() if p.requires_grad]
    gm = torch.autograd.grad(m_loss, parameters, retain_graph=True, allow_unused=True)
    gj = torch.autograd.grad(j_loss, parameters, retain_graph=True, allow_unused=True)

    def real_vector(gradients):
        pieces = []
        for p, g in zip(parameters, gradients):
            g = torch.zeros_like(p) if g is None else g
            if g.is_complex():
                g = torch.view_as_real(g)
            pieces.append(g.detach().cpu().numpy().reshape(-1))
        return np.concatenate(pieces)

    vm, vj = real_vector(gm), real_vector(gj)
    expected_cosine = float(vm @ vj/(np.linalg.norm(vm)*np.linalg.norm(vj)))
    model.invalidate_caches(force=True)
    model.eval()
    next(model.parameters()).grad = torch.full_like(next(model.parameters()), .25)
    state = _state(model)
    report = run_gradient_diagnostics(model, [record], coarse_grid=2, jacobian_loss_weight=.7)
    assert report['complete'] and report['actual_parameter_updates'] == 0
    row = report['rows'][0]
    assert row['jacobian_gradient_status'] == 'QUALIFIED'
    overall = row['gradient_alignment']['total']
    assert overall['moment_gradient_norm'] == pytest.approx(np.linalg.norm(vm), rel=1e-11)
    assert overall['jacobian_gradient_norm'] == pytest.approx(np.linalg.norm(vj), rel=1e-11)
    assert overall['cosine'] == pytest.approx(expected_cosine, abs=1e-11)
    assert overall['combined_gradient_norm'] == pytest.approx(np.linalg.norm(vm+.7*vj), rel=1e-11)
    assert row['gradient_alignment']['shared_encoder']['jacobian_gradient_norm'] > 0.
    assert row['gradient_alignment']['response']['moment_gradient_norm'] > 0.
    json.dumps(report, allow_nan=False)
    _assert_state_preserved(model, state)


def test_complex_inner_product_and_zero_direction_are_not_misreported():
    parameter = torch.nn.Parameter(torch.tensor([0j], dtype=torch.complex128))
    gm = torch.tensor([1+2j], dtype=torch.complex128)
    gj = torch.tensor([3-4j], dtype=torch.complex128)
    report = gradient_alignment([('shared_encoder.spectrum', parameter)], [gm], [gj], jacobian_loss_weight=2.)
    row = report['total']
    assert row['real_inner_product'] == -5.
    assert row['cosine'] == pytest.approx(-1/math.sqrt(5))
    assert row['combined_gradient_norm'] == pytest.approx(math.sqrt(85.))
    zero = gradient_alignment([('response.bias', parameter)], [None], [gj])
    assert zero['total']['cosine'] is None
    assert zero['total']['moment_gradient_tensors'] == 0
    assert zero['shared_encoder']['status'] == 'NO_TRAINABLE_PARAMETERS'
    # Accumulation is in FP64 even for large, finite FP32 gradients.
    large = torch.tensor([1e30], dtype=torch.float32)
    p32 = torch.nn.Parameter(torch.zeros(1))
    finite = gradient_alignment([('response.weight', p32)], [large], [large])
    assert finite['total']['moment_gradient_norm'] == pytest.approx(float(large[0]), rel=1e-12)
    assert finite['total']['cosine'] == pytest.approx(1.)


def test_unqualified_derivative_is_never_read_or_given_a_gradient(model):
    record = copy.deepcopy(synthetic_dataset().train[0])
    record['case']['meta'].update(derivative_valid=False, derivative_dimension=None,
                                derivative_reason='fixture_event')
    record['case']['native_jacobian'] = np.full((10, 10), np.nan)
    report = run_gradient_diagnostics(model, [record], coarse_grid=2)
    assert report['complete']
    row = report['rows'][0]
    assert row['native_jacobian'] is None
    assert row['predicted_jacobian'] is None
    assert row['losses']['jacobian_unweighted'] is None
    assert row['jacobian_gradient_status'] == 'LABEL_UNQUALIFIED'
    assert row['gradient_alignment']['total']['jacobian_gradient_norm'] == 0.
    assert row['gradient_alignment']['total']['cosine'] is None
    json.dumps(report, allow_nan=False)


def test_held_out_gradient_audit_and_duplicate_queries_are_rejected(model):
    record = synthetic_dataset().train[0]
    held_out = dict(record, role='development')
    with pytest.raises(ValueError, match='training queries'):
        run_gradient_diagnostics(model, [held_out], coarse_grid=2)
    with pytest.raises(ValueError, match='distinct'):
        run_path_diagnostics(model, [record, record], coarse_grids=(2,))
    with pytest.raises(ValueError, match='coarse grid'):
        run_path_diagnostics(model, [record], coarse_grids=(3,))


def test_failed_query_retained_and_state_restored(model):
    record = copy.deepcopy(synthetic_dataset().train[0])
    record['case']['native_moments'] = np.full(10, np.nan)
    state = _state(model)
    report = run_path_diagnostics(model, [record], coarse_grids=(2,), diagnostic_nodes=0)
    for result in report['routes'].values():
        assert not result['complete']
        assert len(result['failed_predictions']) == 1
        assert result['rows'][0]['predicted_moments'] is not None
        assert result['source_macro_raw_moment_rmse'] is None
    json.dumps(report, allow_nan=False)
    _assert_state_preserved(model, state)


@pytest.mark.parametrize('objectives', ['M', 'J', 'both', 'J_unavailable'])
def test_last_gradient_releases_graph_and_missing_response_has_no_fake_j(model, monkeypatch, objectives):
    import experiments.henmo_v1plus.diagnostics as diagnostics
    records = synthetic_dataset().train
    if objectives == 'M':
        records = [dict(row, jacobian_weight=0.) for row in records]
    elif objectives == 'J':
        records = [dict(row, moment_weight=0.) for row in records]
    if objectives == 'J_unavailable':
        original_evaluate = model.evaluate_operand
        def without_j(*args, **kwargs):
            return replace(original_evaluate(*args, **kwargs), jacobian=None)
        monkeypatch.setattr(model, 'evaluate_operand', without_j)
    real_gradients = diagnostics._gradients
    flags = []
    def observed_gradients(loss, parameters, *, retain_graph=False):
        if loss is not None and loss.requires_grad:
            flags.append(retain_graph)
        return real_gradients(loss, parameters, retain_graph=retain_graph)
    monkeypatch.setattr(diagnostics, '_gradients', observed_gradients)
    real_predict = diagnostics._model_prediction
    last = []
    def observed_predict(*args, **kwargs):
        # The previous raw prediction must be released before a new encoder
        # forward allocates its graph, even in the two-objective audit.
        assert all(reference() is None for reference in last)
        prediction, timing = real_predict(*args, **kwargs)
        last[:] = [weakref.ref(prediction.moments)]
        if prediction.jacobian is not None:
            last.append(weakref.ref(prediction.jacobian))
        return prediction, timing
    monkeypatch.setattr(diagnostics, '_model_prediction', observed_predict)
    report = run_gradient_diagnostics(model, records, coarse_grid=2)
    assert report['complete']
    assert flags == ([True, False]*2 if objectives == 'both' else [False]*2)
    assert all(reference() is None for reference in last)
    if objectives == 'J_unavailable':
        assert report['derivative_objective_complete'] is False
        assert report['observed_derivative_objectives'] == 0
        assert all(row['jacobian_gradient_status'] == 'RESPONSE_J_UNAVAILABLE' for row in report['rows'])
