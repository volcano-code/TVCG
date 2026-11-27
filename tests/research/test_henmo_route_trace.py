"""Opt-in route evidence must not alter the physical prediction or gradient."""
from dataclasses import replace
import math

import numpy as np
import pytest
import torch

from experiments.henmo_v1plus.diagnostics import _DiagnosticStageTrace
from experiments.henmo_v1plus.fixtures import cube_material, plane_operand
from research.NeuralMoments.models.pcraft_ufno.candidate_projects.quadrann_reproduction.spin_adapter.compact_pino_v21 import PackedPrediction
from research.NeuralMoments.models.pcraft_ufno.candidate_projects.quadrann_reproduction.spin_adapter.henmo_config import HENMOConfig
from research.NeuralMoments.models.pcraft_ufno.candidate_projects.quadrann_reproduction.spin_adapter.henmo_operator import HENMOOperator, blend_moments


@pytest.fixture(autouse=True)
def threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(2)
    yield
    torch.set_num_threads(previous)


@pytest.fixture
def mixed_routes():
    torch.manual_seed(4)
    config = HENMOConfig.cpu_smoke(execution='adaptive', training_execution='adaptive',
        response_zero_init=False, moment_tolerance=1e6, leaf_rule='gl2')
    material, operand = cube_material(16), plane_operand(.43)
    with HENMOOperator(config).double().eval() as model:
        cache = model.encode_geometry(material, np.zeros(10), inference=True)
        probe = model.evaluate_operand(cache, operand, diagnostic_nodes=256)
        # Choose an interior blend location from the actual fixed operator,
        # instead of relying on an accidental threshold for random weights.
        errors = [d['moment_discrepancy'] for d in probe.diagnostics['route_trace']['details']
                  if d['moment_discrepancy'] is not None and d['moment_discrepancy'] > 1e-10]
        assert errors
        model.config = replace(config, moment_tolerance=errors[0]/math.sqrt(2.))
        model.invalidate_caches(force=True)
        yield model, material, operand


def test_route_trace_leaves_moment_jacobian_and_all_parameter_gradients_unchanged(mixed_routes):
    model, material, operand = mixed_routes
    model.train()
    outputs = []
    parameters = tuple(p for p in model.parameters() if p.requires_grad)
    for limit in (0, 256):
        cache = model.encode_geometry(material, np.zeros(10), inference=False)
        prediction = model.evaluate_operand(cache, operand, diagnostic_nodes=limit,
                                            parameter_scales=np.array([2., 3., 4.]))
        loss = prediction.moments.square().mean()+.03*prediction.jacobian.square().mean()
        gradients = torch.autograd.grad(loss, parameters, allow_unused=True)
        outputs.append((prediction, gradients))
    plain, traced = outputs
    assert 'route_trace' not in plain[0].diagnostics
    assert traced[0].diagnostics['route_trace']['enabled']
    torch.testing.assert_close(plain[0].moments, traced[0].moments, rtol=0, atol=0)
    torch.testing.assert_close(plain[0].jacobian, traced[0].jacobian, rtol=0, atol=0)
    for a, b in zip(plain[1], traced[1]):
        if a is None:
            assert b is None
        else:
            torch.testing.assert_close(a, b, rtol=0, atol=0)


def test_bounded_details_keep_full_counts_and_nonoverlapping_frontier_mass(mixed_routes):
    model, material, operand = mixed_routes
    cache = model.encode_geometry(material, np.zeros(10), inference=True)
    full = model.evaluate_operand(cache, operand, diagnostic_nodes=256)
    bounded = model.evaluate_operand(cache, operand, diagnostic_nodes=2)
    a, b = full.diagnostics['route_trace'], bounded.diagnostics['route_trace']
    assert a['applicable'] and not a['details_truncated']
    assert len(b['details']) == 2 and b['details_truncated']
    assert a['all_level_counts'] == b['all_level_counts']
    assert a['frontier_counts'] == b['frontier_counts']
    assert a['frontier_material_mass'] == b['frontier_material_mass']
    assert sum(a['frontier_counts'].values()) == full.diagnostics['selected_coarse_blocks']
    assert sum(a['all_level_counts'].values()) == len(a['details'])
    frontier = [row for row in a['details'] if row['initial_frontier']]
    assert len({row['node_id'] for row in frontier}) == len(frontier)
    assert sum(row['material_mass'] for row in frontier) == pytest.approx(a['selected_frontier_material_mass'])
    assert sum(a['frontier_material_mass'].values()) == pytest.approx(a['selected_frontier_material_mass'])
    # Descendants overlap their parents algebraically; they are never added to
    # the frontier mass denominator as additional physical material.
    assert sum(row['material_mass'] for row in a['details']) > a['selected_frontier_material_mass']
    assert a['all_level_counts'].get('blend', 0) > 0
    torch.testing.assert_close(full.moments, bounded.moments, rtol=0, atol=0)
    torch.testing.assert_close(full.jacobian, bounded.jacobian, rtol=0, atol=0)


def test_recorded_gate_term_and_scaled_max_match_local_chain(mixed_routes):
    model, material, operand = mixed_routes
    cache = model.encode_geometry(material, np.zeros(10), inference=True)
    parameter_scale = np.array([2., 3., 4.])
    prediction = model.evaluate_operand(cache, operand, diagnostic_nodes=256,
                                         parameter_scales=parameter_scale)
    trace = prediction.diagnostics['route_trace']
    maxima = []
    for detail in trace['details']:
        if detail['gate_jacobian'] is None:
            continue
        gate = np.asarray(detail['gate_jacobian'])
        jc, jf = np.asarray(detail['coarse']['jacobian']), np.asarray(detail['fine']['jacobian'])
        actual = np.asarray(detail['output']['jacobian'])
        alpha = detail['alpha']
        np.testing.assert_allclose(gate, actual-((1-alpha)*jc+alpha*jf), atol=2e-15, rtol=1e-12)
        displacement = np.asarray(detail['fine']['moments'])-np.asarray(detail['coarse']['moments'])
        np.testing.assert_allclose(gate, np.outer(displacement, detail['alpha_derivative']), atol=2e-15, rtol=1e-12)
        span = cache.hierarchy.side*cache.hierarchy.pitch
        scale = detail['material_mass']*span**np.array([0,1,1,1,2,2,2,2,2,2])
        maxima.append(float(np.max(np.abs(gate*parameter_scale[None, :]/scale[:, None]))))
    assert maxima and max(maxima) > 0
    assert trace['max_scaled_gate_jacobian'] == pytest.approx(max(maxima))


def test_traced_full_operator_matches_independent_rebuilt_finite_difference(mixed_routes):
    model, material, operand = mixed_routes
    cache = model.encode_geometry(material, np.zeros(10), inference=True)
    prediction = model.evaluate_operand(cache, operand, diagnostic_nodes=256)
    for step in (2e-6, 1e-6):
        plus = model.evaluate_operand(cache, plane_operand(.43, perturbation=[step,0,0]),
                                      with_jacobian=False, diagnostic_nodes=2)
        minus = model.evaluate_operand(cache, plane_operand(.43, perturbation=[-step,0,0]),
                                       with_jacobian=False, diagnostic_nodes=2)
        expected = (plus.moments-minus.moments)/(2*step)
        torch.testing.assert_close(prediction.jacobian[:,0], expected, rtol=6e-5, atol=3e-7)


def test_blend_diagnostic_derivative_matches_independent_live_autograd():
    q = torch.tensor([.37, -.12], dtype=torch.float64, requires_grad=True)
    a = torch.arange(1,11,dtype=torch.float64)/20
    b = a.flip(0)/3
    mc = a*q[0]+b*q[1]
    mp = mc+.2*q[0]**2+.1*q[1]
    mf = mc+q[0]**3+q[1]**2
    coarse = PackedPrediction(mc,torch.stack((a,b),dim=1),{})
    probe = PackedPrediction(mp,torch.stack((a+.4*q[0],b+.1),dim=1),{})
    fine = PackedPrediction(mf,torch.stack((a+3*q[0]**2,b+2*q[1]),dim=1),{})
    result = blend_moments(coarse,fine,probe,torch.ones(10),tolerance=.03,
                           diagnostic_derivatives=True)
    plain = blend_moments(coarse,fine,probe,torch.ones(10),tolerance=.03)
    assert 0 < result.diagnostics['blend_alpha'] < 1
    derivative = torch.stack([torch.autograd.grad(value,q,retain_graph=True)[0] for value in result.moments])
    torch.testing.assert_close(result.jacobian, derivative, atol=1e-12, rtol=1e-12)
    torch.testing.assert_close(result.jacobian, plain.jacobian, atol=0, rtol=0)
    alpha = result.diagnostics['blend_alpha']
    recorded = torch.tensor(result.diagnostics['gate_jacobian'],dtype=torch.float64)
    torch.testing.assert_close(recorded, derivative-((1-alpha)*coarse.jacobian+alpha*fine.jacobian),
                               atol=1e-12,rtol=1e-12)


def test_nonadaptive_trace_is_not_mistaken_for_a_refinement_distribution(mixed_routes):
    model, material, operand = mixed_routes
    model.config = replace(model.config, execution='coarse')
    cache = model.encode_geometry(material, np.zeros(10), inference=True)
    result = model.evaluate_operand(cache, operand, diagnostic_nodes=2)
    trace = result.diagnostics['route_trace']
    assert trace['applicable'] is False
    assert trace['all_level_counts'] == trace['frontier_counts'] == {}
    assert trace['details'] == []
    with pytest.raises(ValueError, match='nonnegative integer'):
        model.evaluate_operand(cache, operand, diagnostic_nodes=True)


def test_diagnostic_stage_timing_supports_nested_scopes_without_double_counting():
    ticks = iter((0., 1., 2., 3.))
    trace = _DiagnosticStageTrace(clock=lambda: next(ticks))
    with trace.stage('material_rule'):
        with trace.stage('response'):
            pass
    phases = trace.report()['phases']
    assert phases['material_rule'] == dict(calls=1,inclusive_s=3.,exclusive_s=2.)
    assert phases['response'] == dict(calls=1,inclusive_s=1.,exclusive_s=1.)
    assert sum(row['exclusive_s'] for row in phases.values()) == 3.
