"""Evaluation cache lifetime and budget contracts; physical tests live with the operator."""
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from experiments.compact_pino_v21.training import runner
from research.NeuralMoments.models.pcraft_ufno.candidate_projects.quadrann_reproduction.spin_adapter.v22_inference_cache import InferenceGeometryCache


def _features(value=1., *, grad=False, inference=True, projected=False):
    table = torch.full((2,4),float(value),requires_grad=grad)
    projection = SimpleNamespace(values=torch.ones(2,6)) if projected else None
    return SimpleNamespace(table=table,projected_context=projection,inference_only=inference)


def test_feature_lru_counts_projection_and_evicts_within_budget():
    cache = InferenceGeometryCache(80)
    first = _features(projected=True)  # 32 + 48 bytes.
    cache['first'] = first
    assert cache.get('first') is first
    assert cache.snapshot()['bytes'] == 80
    cache['second'] = _features(2.)
    assert cache.get('first') is None
    assert cache.snapshot()['evictions'] == 1
    assert cache.snapshot()['bytes'] == 32
    assert cache.snapshot()['peak_bytes'] <= 80
    cache.clear()
    assert cache.snapshot()['entries'] == cache.snapshot()['bytes'] == 0


def test_feature_lru_counts_full_retained_storage_for_views():
    features = _features()
    features.table = torch.ones(100)[:1]
    cache = InferenceGeometryCache(16)
    cache['view'] = features
    assert cache.get('view') is None
    assert cache.snapshot()['bypasses'] == 1


@pytest.mark.parametrize('feature', [_features(grad=True), _features(inference=False)])
def test_training_graph_never_enters_evaluation_cache(feature):
    with pytest.raises(ValueError):
        InferenceGeometryCache(1024)['x'] = feature


def test_zero_budget_and_entry_limit_do_not_retain_excess_features():
    disabled = InferenceGeometryCache(0)
    disabled['x'] = _features()
    assert disabled.snapshot()['bytes'] == disabled.snapshot()['entries'] == 0
    cache = InferenceGeometryCache(1024,max_entries=1)
    cache['a'] = _features(); cache['b'] = _features()
    assert cache.get('a') is None and cache.get('b') is not None


class _EvaluationModel:
    """Counter-only stand-in; does not substitute for real operator equivalence tests."""
    def __init__(self, fail=False):
        self.training = True
        self.encoded = 0
        self.fail = fail

    def train(self, state=True):
        self.training = state

    def eval(self):
        self.train(False)

    def encode_geometry(self, blocks, fixed, *, density, inference):
        assert inference and not self.training
        self.encoded += 1
        return _features(blocks)

    def evaluate_operand(self, encoded, operand, **kwargs):
        if self.fail:
            raise RuntimeError('fixture failure')
        return SimpleNamespace(moments=torch.ones(10,dtype=torch.float64)*encoded.table[0,0],
                               jacobian=None,diagnostics={})

    def efficiency_statistics(self):
        return {'encoded':self.encoded}


def _trainer(monkeypatch, *, efficient=True, budget=1024, fail=False):
    monkeypatch.setattr(runner,'geometry_feature_key',lambda blocks,fixed,density:(blocks,density))
    records=[]
    for index, geometry in enumerate((1,2,1,2,1)):
        records.append(dict(record_id=str(index),jacobian_weight=1.,case=dict(
            blocks=geometry,fixed_moments=np.zeros(10),operand=None,native_moments=np.zeros(10),
            meta=dict(density=1.,derivative_valid=False,editable_mass=1.,length_scale=1.,
                      source_group=str(geometry),chart_id='none'))))
    trainer = runner.Trainer.__new__(runner.Trainer)
    trainer.config = replace(runner.TrainConfig(),efficiency=efficient,evaluation_cache_bytes=budget)
    trainer.data = SimpleNamespace(development=records)
    trainer.mode = 'smoke'
    trainer.model = _EvaluationModel(fail)
    return trainer


def test_development_reuses_encoding_preserves_query_order_and_ends_lifetime(monkeypatch):
    trainer = _trainer(monkeypatch)
    rng = torch.get_rng_state().clone()
    first = trainer.evaluate_development()
    assert trainer.model.encoded == 2
    assert [row['record_id'] for row in first['rows']] == ['0','1','2','3','4']
    assert first['efficiency']['evaluation_features']['hits'] == 3
    assert trainer.model.training and torch.equal(torch.get_rng_state(),rng)
    second = trainer.evaluate_development()
    assert trainer.model.encoded == 4  # A later checkpoint/pass must re-encode.
    assert first['rows'] == second['rows']


@pytest.mark.parametrize('efficient,budget', [(False,1024),(True,0)])
def test_reference_or_disabled_cache_encodes_each_query(monkeypatch,efficient,budget):
    trainer = _trainer(monkeypatch,efficient=efficient,budget=budget)
    result = trainer.evaluate_development()
    assert trainer.model.encoded == 5
    assert result['efficiency']['evaluation_features']['bytes'] == 0


def test_failed_evaluation_restores_model_mode_and_rng(monkeypatch):
    trainer = _trainer(monkeypatch,fail=True)
    before = torch.get_rng_state().clone()
    with pytest.raises(RuntimeError,match='fixture failure'):
        trainer.evaluate_development()
    assert trainer.model.training and torch.equal(before,torch.get_rng_state())


def test_persistent_cache_rejected_for_training_forward(monkeypatch):
    trainer = _trainer(monkeypatch)
    with pytest.raises(ValueError,match='fixed-weight evaluation'):
        trainer._predictions(trainer.data.development,geometry_cache=InferenceGeometryCache())


@pytest.mark.parametrize('change', [dict(efficiency=1),dict(static_cache_bytes=-1),
                                   dict(query_cache_bytes=True),dict(evaluation_cache_bytes=1.5)])
def test_cache_settings_are_explicit_reproducible_config(change):
    with pytest.raises(ValueError,match='cache byte budgets'):
        replace(runner.TrainConfig(),**change).validate()
