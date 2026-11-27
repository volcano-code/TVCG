"""Whole-operator checks, including a genuine shared P-CRAFT M+J backward."""
from dataclasses import replace
import numpy as np
import pytest
import torch

from research.NeuralMoments.models.pcraft_ufno.candidate_projects.quadrann_reproduction.spin_adapter.henmo_config import HENMOConfig
from research.NeuralMoments.models.pcraft_ufno.candidate_projects.quadrann_reproduction.spin_adapter.henmo_operator import HENMOOperator, blend_moments
from research.NeuralMoments.models.pcraft_ufno.candidate_projects.quadrann_reproduction.spin_adapter.compact_pino_v21 import PackedPrediction
from experiments.henmo_v1plus.fixtures import cube_material, plane_operand, synthetic_dataset


@pytest.fixture(autouse=True)
def threads():
    old = torch.get_num_threads()
    torch.set_num_threads(2)
    yield
    torch.set_num_threads(old)


def test_configuration_decouples_resolution_and_has_no_fake_exact_flag():
    for side in (512,1024):
        config = HENMOConfig(native_material_resolution=side)
        assert config.context_grids == (64,128) and config.leaf_grid == 64
    with pytest.raises(ValueError,match='native acceptance'):
        HENMOConfig(require_native_acceptance=False)
    with pytest.raises(ValueError,match='ten-moment'):
        HENMOConfig(max_coarse_atoms=1)


def test_blend_chain_contains_gate_derivative_and_matches_independent_ad():
    q = torch.tensor(.37,dtype=torch.float64,requires_grad=True)
    a = torch.arange(1,11,dtype=torch.float64)/20
    mc = a*q
    mp = mc+.2*q*q
    mf = a*q+q**3
    coarse = PackedPrediction(mc,a[:,None],{})
    probe = PackedPrediction(mp,(a+.4*q)[:,None],{})
    fine = PackedPrediction(mf,(a+3*q*q)[:,None],{})
    result = blend_moments(coarse,fine,probe,torch.ones(10,dtype=torch.float64),tolerance=.06)
    assert 0 < result.diagnostics['blend_alpha'] < 1
    expected = torch.stack([torch.autograd.grad(row,q,retain_graph=True)[0] for row in result.moments])
    torch.testing.assert_close(result.jacobian[:,0],expected,rtol=1e-12,atol=1e-12)


@pytest.mark.parametrize('execution',['coarse','adaptive','leaf'])
def test_finite_operand_same_side_complement_and_rebuilt_finite_difference(execution):
    torch.manual_seed(12)
    config = HENMOConfig.cpu_smoke(execution=execution,response_zero_init=False,
                                   moment_tolerance=.05,leaf_rule='gl2')
    with HENMOOperator(config).double().eval() as model:
        material = cube_material(16)
        fixed = np.array([.1,0,0,0,0,0,0,0,0,0])
        cache = model.encode_geometry(material,fixed,inference=True)
        q = plane_operand(.43)
        prediction = model.evaluate_operand(cache,q)
        other = model.evaluate_operand(cache,replace(q,operation='difference'))
        np.testing.assert_allclose((prediction.moments+other.moments).detach(),2*fixed+material.moments(),atol=2e-11)
        assert prediction.jacobian is not None
        np.testing.assert_allclose((prediction.jacobian+other.jacobian).detach(),0,atol=2e-10)
        h = 2e-6
        plus = model.evaluate_operand(cache,plane_operand(.43,perturbation=[h,0,0]),with_jacobian=False)
        minus = model.evaluate_operand(cache,plane_operand(.43,perturbation=[-h,0,0]),with_jacobian=False)
        np.testing.assert_allclose(((plus.moments-minus.moments)/(2*h)).detach(),
                                   prediction.jacobian[:,0].detach(),rtol=5e-5,atol=2e-7)
        assert prediction.diagnostics['error_indicator_is_certificate'] is False


def test_actual_shared_encoder_attention_and_response_receive_jacobian_gradient():
    torch.manual_seed(82)
    with HENMOOperator(HENMOConfig.cpu_smoke(response_zero_init=False)).double() as model:
        material = cube_material(16)
        cache = model.encode_geometry(material,np.zeros(10))
        prediction = model.evaluate_operand(cache,plane_operand(.43))
        assert prediction.jacobian is not None and prediction.jacobian.requires_grad
        loss = prediction.jacobian.square().mean()
        loss.backward()
        def gradient(prefix):
            return sum(float(p.grad.abs().sum()) for name,p in model.named_parameters()
                       if name.startswith(prefix) and p.grad is not None)
        assert gradient('shared_encoder.') > 0
        assert gradient('local_geometry.') > 0
        assert gradient('response.cut_key.') > 0
        assert gradient('response.output.') > 0
        with pytest.raises(ValueError,match='consumed'):
            model.evaluate_operand(cache,plane_operand(.44))


def test_coarse_frontier_batches_actual_network_calls_and_supports_inference_mode():
    config = HENMOConfig.cpu_smoke(execution='coarse',response_zero_init=False)
    with HENMOOperator(config).double().eval() as model:
        with torch.inference_mode():
            cache = model.encode_geometry(cube_material(16),np.zeros(10),inference=True)
            prediction = model.evaluate_operand(cache,plane_operand(.43))
            value_only = model.evaluate_operand(cache,plane_operand(.43),with_jacobian=False)
        diagnostics = prediction.diagnostics
        assert diagnostics['coarse_evaluations'] > 1
        assert diagnostics['neural_network_calls'] == 1
        assert diagnostics['boundary_batch_groups'] == 1
        assert diagnostics['active_atoms'] > 0
        torch.testing.assert_close(value_only.moments,prediction.moments)
        assert value_only.diagnostics['solver_status'] == 'MOMENT_ONLY_REQUIRES_NATIVE_ACCEPTANCE'
        assert value_only.jacobian is None


def test_accepted_coarse_parent_uses_cheap_children_without_full_leaf_expansion():
    config = HENMOConfig.cpu_smoke(execution='adaptive',moment_tolerance=1.e6,
                                   jacobian_tolerance=1.e6)
    with HENMOOperator(config).double().eval() as model:
        cache = model.encode_geometry(cube_material(16),np.zeros(10),inference=True)
        result = model.evaluate_operand(cache,plane_operand(.43))
        assert result.diagnostics['coarse_evaluations'] > result.diagnostics['selected_coarse_blocks']
        assert result.diagnostics['deterministic_leaf_evaluations'] == 0
        assert result.diagnostics['refined_nodes'] == 0
        assert result.diagnostics['neural_network_calls'] == 2
        assert result.diagnostics['boundary_batch_groups'] == 2


def test_coarse_support_prevents_distance_gap_from_silencing_every_atom():
    from research.NeuralMoments.models.pcraft_ufno.candidate_projects.quadrann_reproduction.spin_adapter.v21_response import prepare_measure,prepare_stencil,prepare_surface
    config = HENMOConfig.cpu_smoke(coarse_grid=1,training_coarse_grids=(1,),execution='coarse')
    with HENMOOperator(config).double() as model:
        cache = model.encode_geometry(cube_material(16),np.zeros(10))
        measure = cache.hierarchy.measure(cache.hierarchy.root,max_atoms=config.max_coarse_atoms)
        # Search only the static cubature's x gaps. This establishes the exact
        # old failure without relying on one particular compression ordering.
        x = np.unique(measure.nodes[:,0])
        ends = np.r_[0.,x,1.]
        gaps = ends[1:]-ends[:-1]
        k = int(np.argmax(gaps))
        cut = float((ends[k]+ends[k+1])/2.)
        assert gaps[k] > 2*cache.hierarchy.pitch
        operand = plane_operand(cut)
        surface = prepare_surface(operand.vertices,operand.faces)
        narrow = prepare_stencil(prepare_measure(measure.nodes,measure.weights),surface,
                                 bandwidth=cache.hierarchy.pitch)
        assert len(narrow.nodes) == 0
        prediction = model.evaluate_operand(cache,operand)
        assert prediction.diagnostics['active_atoms'] > 0
        prediction.moments[0].backward()
        assert model.response.output.weight.grad.abs().sum() > 0


def test_encoder_parameter_and_buffer_mutation_invalidates_cached_geometry():
    with HENMOOperator(HENMOConfig.cpu_smoke(training_scope='response')).eval() as model:
        material = cube_material(16)
        cache = model.encode_geometry(material,np.zeros(10),inference=True)
        with torch.no_grad():
            next(model.shared_encoder.parameters()).add_(.001)
        with pytest.raises(ValueError,match='stale'):
            model.evaluate_operand(cache,plane_operand())
        cache = model.encode_geometry(material,np.zeros(10),inference=True)
        with torch.no_grad():
            next(model.shared_encoder.buffers()).add_(.001)
        with pytest.raises(ValueError,match='stale'):
            model.evaluate_operand(cache,plane_operand())


def test_deterministic_model_bypasses_encoder_and_matches_fine_baseline():
    from research.NeuralMoments.models.pcraft_ufno.candidate_projects.quadrann_reproduction.spin_adapter.henmo_hierarchy import evaluate_deterministic
    from research.NeuralMoments.models.pcraft_ufno.candidate_projects.quadrann_reproduction.spin_adapter.v21_response import prepare_surface
    with HENMOOperator(HENMOConfig.cpu_smoke(response_path='deterministic',leaf_rule='gl2')).eval() as model:
        material = cube_material(16)
        cache = model.encode_geometry(material,np.zeros(10),inference=True)
        operand = plane_operand()
        prediction = model.evaluate_operand(cache,operand)
        reference = evaluate_deterministic(cache.hierarchy,prepare_surface(operand.vertices,operand.faces),
                    material.pitch,vertex_velocity=operand.vertex_velocity,full_cell_rule='gl2')
        torch.testing.assert_close(prediction.moments,reference.moments,rtol=1e-13,atol=1e-13)
        torch.testing.assert_close(prediction.jacobian,reference.jacobian,rtol=1e-12,atol=1e-12)
        assert model.efficiency_statistics()['encoder_calls'] == 0


def test_real_model_two_updates_and_exact_next_step_resume(tmp_path):
    from experiments.henmo_v1plus.dataset import dataset_binding,model_contract
    from experiments.henmo_v1plus.training import HENMOTrainer,HENMOTrainingConfig
    torch.manual_seed(11)
    data = synthetic_dataset()
    config = HENMOConfig.cpu_smoke()
    model = HENMOOperator(config).double()
    trainer = HENMOTrainer(model,HENMOTrainingConfig(max_steps=3),dataset_binding(data,model_contract(model)),records=data.train)
    first = trainer.step()
    assert first['committed'] and trainer.step_count == 1
    path = tmp_path/'state.pt'
    trainer.save_checkpoint(path)
    second = trainer.step()
    assert second['committed']
    expected = {key:value.clone() for key,value in model.state_dict().items()}
    restored = HENMOOperator(config).double()
    resumed = HENMOTrainer(restored,trainer.config,trainer.binding,records=data.train)
    resumed.load_checkpoint(path)
    again = resumed.step()
    assert again['committed'] and again['record_ids'] == second['record_ids']
    for key,value in restored.state_dict().items():
        torch.testing.assert_close(value,expected[key],rtol=0,atol=0)
    model.close(); restored.close()
