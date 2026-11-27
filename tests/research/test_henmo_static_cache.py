"""Static preparation reuse must never reuse learned values or AD graphs."""
from dataclasses import replace

import numpy as np
import pytest
import torch

from experiments.henmo_v1plus.fixtures import cube_material, plane_operand, hard_cube_moments
from research.NeuralMoments.models.pcraft_ufno.candidate_projects.quadrann_reproduction.spin_adapter import henmo_geometry as geometry
from research.NeuralMoments.models.pcraft_ufno.candidate_projects.quadrann_reproduction.spin_adapter.henmo_config import HENMOConfig
from research.NeuralMoments.models.pcraft_ufno.candidate_projects.quadrann_reproduction.spin_adapter.henmo_operator import HENMOOperator


@pytest.fixture(autouse=True)
def threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(2)
    yield
    torch.set_num_threads(previous)


def _tree(model, material=None):
    return geometry.make_hierarchy(cube_material(16) if material is None else material,
                                   model.config)


def test_cache_matches_uncached_value_and_skips_static_preparation(monkeypatch):
    torch.manual_seed(818)
    with HENMOOperator(HENMOConfig.cpu_smoke()).double() as model:
        tree, fixed = _tree(model), np.zeros(10)
        cache = geometry.StaticGeometryInputCache()
        calls = {'aggregate':0, 'image':0, 'encoder':0}
        aggregate, image = geometry.aggregate_grid, geometry.encoder_grid
        forward = model.shared_encoder.pino_encoder.forward_with_map
        def aggregate_count(*args, **kwargs):
            calls['aggregate'] += 1
            return aggregate(*args, **kwargs)
        def image_count(*args, **kwargs):
            calls['image'] += 1
            return image(*args, **kwargs)
        def forward_count(*args, **kwargs):
            calls['encoder'] += 1
            return forward(*args, **kwargs)
        monkeypatch.setattr(geometry,'aggregate_grid',aggregate_count)
        monkeypatch.setattr(geometry,'encoder_grid',image_count)
        monkeypatch.setattr(model.shared_encoder.pino_encoder,'forward_with_map',forward_count)
        expected = geometry.encode_table(model,tree,fixed,1.)
        first = geometry.encode_table(model,tree,fixed,1.,static_cache=cache)
        counts_after_miss = calls.copy()
        hit = geometry.encode_table(model,tree,fixed,1.,static_cache=cache)
        torch.testing.assert_close(first,expected,rtol=0,atol=0)
        torch.testing.assert_close(hit,expected,rtol=0,atol=0)
        assert calls['aggregate'] == counts_after_miss['aggregate']
        assert calls['image'] == counts_after_miss['image']
        assert calls['encoder'] == counts_after_miss['encoder']+len(model.config.context_grids)
        assert cache.snapshot()['hits'] == 1
        assert hit.grad_fn is not first.grad_fn


def test_two_real_updates_match_uncached_and_rebuild_learned_graphs():
    torch.manual_seed(971)
    config = HENMOConfig.cpu_smoke()
    with HENMOOperator(config).double() as cached, HENMOOperator(config).double() as reference:
        reference.load_state_dict(cached.state_dict())
        tree_a, tree_b, fixed = _tree(cached), _tree(reference), np.zeros(10)
        cache = geometry.StaticGeometryInputCache()
        optimizers = [torch.optim.Adam(m.parameters(),lr=3e-5) for m in (cached,reference)]
        outputs = []
        for _ in range(2):
            for model, tree, optimizer, optional in zip((cached,reference),(tree_a,tree_b),
                                                       optimizers,(cache,None)):
                optimizer.zero_grad(set_to_none=True)
                table = geometry.encode_table(model,tree,fixed,1.,static_cache=optional)
                outputs.append(table.detach().clone())
                loss = table.square().mean()+.01*table.sum()
                loss.backward()
                assert any(p.grad is not None and bool(p.grad.abs().sum() > 0)
                           for p in model.shared_encoder.parameters())
                optimizer.step()
            torch.testing.assert_close(outputs[-2],outputs[-1],rtol=0,atol=0)
            for name,value in cached.state_dict().items():
                torch.testing.assert_close(value,reference.state_dict()[name],rtol=0,atol=0)
        assert not torch.equal(outputs[0],outputs[2])
        assert cache.snapshot()['hits'] == 1
        assert cache.snapshot()['misses'] == 1
        assert all(not t.requires_grad and t.grad_fn is None
                   for entry,_ in cache._entries.values() for t in entry.tensors)


def test_inference_prepared_cache_is_safe_for_later_training_backward():
    with HENMOOperator(HENMOConfig.cpu_smoke()).double() as model:
        tree, fixed, cache = _tree(model), np.zeros(10), geometry.StaticGeometryInputCache()
        with torch.inference_mode():
            inputs = cache.prepare(model,tree,fixed,1.)
            geometry.encode_table(model,tree,fixed,1.,static_cache=cache)
        assert all(not t.is_inference() for t in inputs.tensors)
        geometry.encode_table(model,tree,fixed,1.,static_cache=cache).square().sum().backward()
        assert any(p.grad is not None and bool(p.grad.abs().sum() > 0)
                   for p in model.shared_encoder.parameters())


def test_material_identity_reuses_rebuilt_tree_and_fixed_or_grid_changes_miss():
    with HENMOOperator(HENMOConfig.cpu_smoke()).double() as model:
        material = cube_material(16)
        first, second = _tree(model,material), _tree(model,material)
        assert first is not second and first.identity == second.identity
        cache, fixed = geometry.StaticGeometryInputCache(), np.zeros(10)
        a = cache.prepare(model,first,fixed,1.)
        assert cache.prepare(model,second,fixed,1.) is a
        fixed[0] = .1
        assert cache.prepare(model,second,fixed,1.) is not a
        model.config = replace(model.config,feature_grid=4)
        assert cache.prepare(model,second,fixed,1.).active.shape != a.active.shape
        assert cache.snapshot()['misses'] == 3
        assert cache.snapshot()['hits'] == 1


def test_cache_checks_material_replacement_and_tensor_mutation_on_hit():
    with HENMOOperator(HENMOConfig.cpu_smoke()).double() as model:
        material = cube_material(16)
        tree, fixed, cache = _tree(model,material), np.zeros(10), geometry.StaticGeometryInputCache()
        inputs = cache.prepare(model,tree,fixed,1.)
        with torch.no_grad():
            inputs.features.add_(.1)
        with pytest.raises(ValueError,match='cache tensor was modified'):
            cache.prepare(model,tree,fixed,1.)
        cache.clear()
        cache.prepare(model,tree,fixed,1.)
        material.full_runs = np.frombuffer(material.full_runs.tobytes(),dtype=material.full_runs.dtype).reshape(material.full_runs.shape)
        with pytest.raises(ValueError,match='source material was replaced'):
            cache.prepare(model,tree,fixed,1.)


def test_byte_bound_lru_and_disabled_cache():
    with HENMOOperator(HENMOConfig.cpu_smoke()).double() as model:
        tree, a, b, c = _tree(model), np.zeros(10), np.zeros(10), np.zeros(10)
        b[0],c[0] = .1,.2
        disabled = geometry.StaticGeometryInputCache(0)
        size = disabled.prepare(model,tree,a,1.).nbytes
        disabled.prepare(model,tree,a,1.)
        assert disabled.snapshot()['entries'] == disabled.snapshot()['bytes'] == 0
        assert disabled.snapshot()['misses'] == disabled.snapshot()['oversized'] == 2
        cache = geometry.StaticGeometryInputCache(2*size)
        ia = cache.prepare(model,tree,a,1.)
        ib = cache.prepare(model,tree,b,1.)
        assert cache.prepare(model,tree,a,1.) is ia  # a is most recently used.
        cache.prepare(model,tree,c,1.)               # b must be evicted.
        assert cache.prepare(model,tree,a,1.) is ia
        assert cache.prepare(model,tree,b,1.) is not ib
        assert cache.snapshot()['bytes'] <= cache.max_bytes
        assert cache.snapshot()['entries'] == 2
        assert cache.snapshot()['evictions'] == 2
        cache.clear()
        assert cache.snapshot()['entries'] == cache.snapshot()['bytes'] == 0


def test_cache_dtype_and_density_keys_are_distinct():
    with HENMOOperator(HENMOConfig.cpu_smoke()) as model:
        tree, fixed, cache = _tree(model), np.zeros(10), geometry.StaticGeometryInputCache()
        a = cache.prepare(model,tree,fixed,1.)
        b = cache.prepare(model,tree,fixed,2.)
        assert a is not b and not torch.equal(a.images[0],b.images[0])
        model.double()
        c = cache.prepare(model,tree,fixed,1.)
        assert a.features.dtype == torch.float32 and c.features.dtype == torch.float64
        assert cache.snapshot()['misses'] == 3


@pytest.mark.parametrize('budget',[-1,1.5,True])
def test_invalid_cache_budget_is_rejected(budget):
    with pytest.raises(ValueError,match='nonnegative integer'):
        geometry.StaticGeometryInputCache(budget)


def test_whole_operator_moment_jacobian_and_two_adam_updates_match_cache_off():
    torch.manual_seed(961)
    config = HENMOConfig.cpu_smoke(response_zero_init=False)
    with HENMOOperator(config).double() as cached, HENMOOperator(config,static_input_cache_bytes=0).double() as reference:
        reference.load_state_dict(cached.state_dict())
        material = cube_material(16)
        fixed = .02*np.array([1,.25,.25,.25,.0625,.0625,.0625,.0625,.0625,.0625])
        optimizers = [torch.optim.Adam(m.parameters(),lr=3e-5) for m in (cached,reference)]
        initial = {name: value.detach().clone() for name,value in cached.state_dict().items()}
        for offset in (.43,.61):
            predictions = []
            target_m,target_j = hard_cube_moments(offset)
            for model, optimizer in zip((cached,reference),optimizers):
                optimizer.zero_grad(set_to_none=True)
                encoded = model.encode_geometry(material,fixed)
                result = model.evaluate_operand(encoded,plane_operand(offset),coarse_grid=2)
                predictions.append((result.moments.detach().clone(),result.jacobian.detach().clone()))
                loss = ((result.moments-result.moments.new_tensor(target_m+fixed)).square().mean()
                       +.1*(result.jacobian-result.jacobian.new_tensor(target_j)).square().mean())
                loss.backward()
                assert any(p.grad is not None and bool(p.grad.abs().sum() > 0)
                           for p in model.shared_encoder.parameters())
                optimizer.step()
                with pytest.raises(ValueError,match='stale'):
                    model.evaluate_operand(encoded,plane_operand(offset))
            for a,b in zip(predictions[0],predictions[1]):
                torch.testing.assert_close(a,b,rtol=0,atol=0)
            for name,value in cached.state_dict().items():
                torch.testing.assert_close(value,reference.state_dict()[name],rtol=0,atol=0)
            states = [optimizer.state_dict()['state'] for optimizer in optimizers]
            assert states[0].keys() == states[1].keys()
            for key,state in states[0].items():
                for name,value in state.items():
                    torch.testing.assert_close(value,states[1][key][name],rtol=0,atol=0)
        stats = cached.efficiency_statistics()
        assert stats['encoder_calls'] == 2
        assert stats['static_input_cache']['misses'] == stats['static_input_cache']['hits'] == 1
        assert reference.efficiency_statistics()['static_input_cache']['entries'] == 0
        assert any(not torch.equal(value,initial[name]) for name,value in cached.state_dict().items())


def test_alternating_materials_reuse_static_inputs_after_hierarchy_rebuild():
    from research.NeuralMoments.models.pcraft_ufno.candidate_projects.quadrann_reproduction.spin_adapter.v21_material import prepare_rle_material
    config = HENMOConfig.cpu_smoke(response_zero_init=False,execution='coarse')
    material_a = cube_material(16)
    columns = np.array([[x,y,0,16] for x in range(16) for y in range(8)],np.int32)
    material_b = prepare_rle_material(columns,np.zeros(3),1./16,16)
    assert material_a.identity != material_b.identity
    with HENMOOperator(config).double().eval() as cached, HENMOOperator(config,static_input_cache_bytes=0).double().eval() as reference:
        reference.load_state_dict(cached.state_dict())
        values = []
        with torch.inference_mode():
            for material in (material_a,material_b,material_a):
                predictions = []
                for model in (cached,reference):
                    encoded = model.encode_geometry(material,np.zeros(10),inference=True)
                    prediction = model.evaluate_operand(encoded,plane_operand(.43))
                    predictions.append(prediction)
                torch.testing.assert_close(predictions[0].moments,predictions[1].moments,rtol=0,atol=0)
                torch.testing.assert_close(predictions[0].jacobian,predictions[1].jacobian,rtol=0,atol=0)
                values.append(predictions[0].moments.clone())
        assert not torch.equal(values[0],values[1])
        torch.testing.assert_close(values[0],values[2],rtol=0,atol=0)
        stats = cached.efficiency_statistics()
        assert stats['hierarchy_builds'] == stats['encoder_calls'] == 3
        assert stats['static_input_cache']['misses'] == stats['static_input_cache']['entries'] == 2
        assert stats['static_input_cache']['hits'] == 1
        cached.clear_efficiency_caches()
        assert cached.efficiency_statistics()['static_input_cache']['bytes'] == 0
        assert cached.efficiency_statistics()['hierarchy'] is None


def test_operator_dtype_change_evicts_static_device_inputs():
    with HENMOOperator(HENMOConfig.cpu_smoke()).eval() as model:
        material = cube_material(16)
        model.encode_geometry(material,np.zeros(10),inference=True)
        assert model.efficiency_statistics()['static_input_cache']['bytes'] > 0
        model.double()
        assert model.efficiency_statistics()['static_input_cache']['bytes'] == 0
        model.encode_geometry(material,np.zeros(10),inference=True)
        assert model.efficiency_statistics()['static_input_cache']['entries'] == 1
        inputs,_ = next(iter(model._static_input_cache._entries.values()))
        assert inputs.features.dtype == torch.float64


def test_learned_geometry_versions_track_encoding_semantics_not_query_paths():
    with HENMOOperator(HENMOConfig.cpu_smoke()).double().eval() as model:
        material = cube_material(16)
        encoded = model.encode_geometry(material,np.zeros(10),inference=True)
        original_versions = encoded.versions
        model.config = replace(model.config,coarse_grid=4,execution='coarse')
        assert geometry.geometry_versions(model) == original_versions
        assert model.encode_geometry(material,np.zeros(10),inference=True).table is encoded.table
        model.config = replace(model.config,feature_grid=4)
        assert geometry.geometry_versions(model) != original_versions
        with pytest.raises(ValueError,match='stale'):
            model.evaluate_operand(encoded,plane_operand(.43))
        replacement = model.encode_geometry(material,np.zeros(10),inference=True)
        assert replacement.table.shape[0] == 4**3
        assert replacement.table is not encoded.table
