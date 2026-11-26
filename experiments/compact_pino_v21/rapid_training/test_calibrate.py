"""Small calibration fixtures: isolation, real gradients, no optimization."""
from dataclasses import asdict
import json
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from experiments.compact_pino_v21.rapid_training import calibrate
from experiments.compact_pino_v21.training.admission import canonical_digest
from experiments.compact_pino_v21.training.runner import CHECKPOINT_SCHEMA, TrainConfig
from experiments.compact_pino_v21.training.test_residual_matching import (
    NATIVE, DELTA, COEFFICIENTS, linearized_target,
)
from research.NeuralMoments.models.pcraft_ufno.candidate_projects.quadrann_reproduction.spin_adapter.v21_response import LocalRetention


def record(rid='train-a', source='source-a', *, qualified=True):
    return dict(record_id=rid, role='train', loader_eligible=True, case_sha256='a'*64,
        moment_weight=1., jacobian_weight=1., residual_weight=1., residual_valid=True,
        residual_target=linearized_target(),
        case=dict(meta=dict(source_group=source, geometry_id=source+'-geometry', query_id=rid,
                            density=1., editable_mass=1., length_scale=1., derivative_valid=qualified),
                  native_moments=NATIVE.copy(), coefficients=COEFFICIENTS.copy(),
                  native_jacobian=np.zeros((10, 3)), parameter_scales=np.ones(3),
                  blocks=SimpleNamespace(pitch=.1), fixed_moments=np.zeros(10), operand=rid))


class TinyModel(torch.nn.Module):
    """Only a differentiable software fixture; no geometry accuracy claim."""
    def __init__(self, *, response_type='bounded_cubic', response_zero_init=False, **kwargs):
        super().__init__()
        self.shared_encoder = torch.nn.Linear(3, 4).double()
        self.geometry_attention = torch.nn.Linear(4, 4).double()
        self.response = LocalRetention(4, 7, response_type=response_type, zero_init=response_zero_init).double()
        self.register_buffer('points', torch.tensor([[-.3, .1, .2], [.1, -.2, .3], [.2, .2, -.1]], dtype=torch.float64))
        self.register_buffer('basis', torch.linspace(.001, .03, 30, dtype=torch.float64).reshape(3, 10))
        self.register_buffer('baseline', torch.tensor(NATIVE+DELTA, dtype=torch.float64))
        for module in (self.shared_encoder, self.geometry_attention):
            module.requires_grad_(False)

    def train(self, mode=True):
        super().train(mode)
        self.shared_encoder.eval()
        self.geometry_attention.eval()
        return self

    def encode_geometry(self, blocks, fixed_moments, density=1.):
        with torch.no_grad():
            return self.geometry_attention(self.shared_encoder(self.points))

    def evaluate_operand(self, encoded, operand, *, with_jacobian, bandwidth, chunk_size):
        distance = self.points[:, 0]
        value, slope = self.response.retention_and_slope(self.points, distance, encoded)
        moments = self.baseline+self.basis.T@(value-.5)
        jacobian = self.basis.T@(slope[:, None]*self.points) if with_jacobian else None
        return SimpleNamespace(moments=moments, jacobian=jacobian)

    def clear_efficiency_caches(self):
        pass

    def efficiency_statistics(self):
        return {'software_fixture': True}


def prediction(model, row):
    case = row['case']
    encoded = model.encode_geometry(case['blocks'], case['fixed_moments'])
    return model.evaluate_operand(encoded, case['operand'],
        with_jacobian=case['meta']['derivative_valid'], bandwidth=.1, chunk_size=16)


def cohort(data, rows=None):
    return dict(schema=calibrate.COHORT_SCHEMA, training_manifest_sha256=data.manifest_sha256,
                records=[dict(record_id=row['record_id']) for row in (data.train if rows is None else rows)])


def test_source_round_robin_and_train_only_membership():
    rows = [record('a'+str(i), 'source-a') for i in range(8)]
    rows += [record('b', 'source-b'), record('c', 'source-c')]
    data = SimpleNamespace(train=rows, manifest_sha256='b'*64)
    selected, receipt = calibrate.select_training_records(data, cohort(data), max_records=4, seed=17)
    assert [row['case']['meta']['source_group'] for row in selected[:3]] == ['source-a', 'source-b', 'source-c']
    assert receipt['unique_records'] == 4 and len(receipt['source_groups']) == 3
    repeated, _ = calibrate.select_training_records(data, cohort(data), max_records=4, seed=17)
    assert [row['record_id'] for row in repeated] == [row['record_id'] for row in selected]
    with pytest.raises(ValueError, match='every eligible'):
        calibrate.select_training_records(data, cohort(data), max_records=2)
    packet = cohort(data)
    packet['records'].append(dict(record_id='held-out-development'))
    with pytest.raises(ValueError, match='only admitted train'):
        calibrate.select_training_records(data, packet)


def test_no_physics_record_is_excluded_and_identity_mismatch_is_rejected():
    rows = [record('valid'), record('invalid')]
    rows[1]['residual_valid'] = False
    data = SimpleNamespace(train=rows, manifest_sha256='b'*64)
    selected, receipt = calibrate.select_training_records(data, cohort(data))
    assert [row['record_id'] for row in selected] == ['valid']
    assert receipt['excluded'][0]['record_id'] == 'invalid'
    packet = cohort(data)
    packet['training_manifest_sha256'] = 'c'*64
    with pytest.raises(ValueError, match='manifest SHA256'):
        calibrate.select_training_records(data, packet)


@pytest.mark.parametrize('qualified', [False, True])
def test_probe_reports_real_independent_gradients_without_updates(qualified):
    torch.manual_seed(14)
    model = TinyModel(response_type='monotone_warp', response_zero_init=True).train()
    row = record(qualified=qualified)
    before = {name: tensor.clone() for name, tensor in model.state_dict().items()}
    pred = prediction(model, row)
    report = calibrate.probe_terms([pred], [row], model)
    assert report['parameter_updates'] == 0
    assert report['gradient_norms']['M'] > 0
    assert report['gradient_norms']['R'] > 0
    assert report['gradient_norms']['C'] > 0
    assert report['terms']['J']['denominator'] == (1. if qualified else 0.)
    if not qualified:
        assert report['gradient_norms']['J'] == 0.
        assert report['gradient_cosines']['M_J'] is None
        assert report['terms']['J']['qualification_reasons'] == ['LABEL_UNQUALIFIED']
    else:
        assert report['gradient_norms']['J'] > 0
        assert -1. <= report['gradient_cosines']['M_J'] <= 1.
    for name, tensor in model.state_dict().items():
        assert torch.equal(tensor, before[name])
    assert all(parameter.grad is None for parameter in model.parameters())
    # Check the M norm independently against its own explicit ten-component
    # mean square; this detects accidentally probing the total joint gradient.
    loss = (pred.moments-torch.from_numpy(NATIVE)).square().mean()
    grads = torch.autograd.grad(loss, tuple(model.response.parameters()), allow_unused=True)
    expected_norm = torch.sqrt(sum(g.square().sum() for g in grads if g is not None)).item()
    assert report['gradient_norms']['M'] == pytest.approx(expected_norm, rel=1e-12)


def test_term_denominators_keep_positive_membership_separate():
    model = TinyModel(response_zero_init=True).train()
    rows = [record('with-j'), record('without-j', qualified=False)]
    rows[0].update(moment_weight=2., jacobian_weight=7., residual_weight=3.)
    rows[1].update(moment_weight=5., jacobian_weight=11., residual_weight=13.)
    result = calibrate.probe_terms([prediction(model, row) for row in rows], rows, model)
    assert {key: value['denominator'] for key, value in result['terms'].items()} == dict(M=7., J=7., R=16., C=16.)
    assert result['terms']['J']['effective_queries'] == 1


def calibration_fixture(tmp_path, monkeypatch):
    manifest = tmp_path/'manifest.json'
    manifest.write_text('{"software_fixture":true}\n')
    rows = [record('a', 'source-a'), record('b', 'source-b', qualified=False)]
    data = SimpleNamespace(train=rows, path=manifest, manifest_sha256=calibrate.sha256(manifest), scope='synthetic')
    packet_path = tmp_path/'cohort.json'
    packet_path.write_text(json.dumps(cohort(data)))
    calls = []
    def loader(path, *, mode, cache_directory=None):
        calls.append((str(path), mode))
        return data
    monkeypatch.setattr(calibrate, 'load_training_manifest', loader)
    monkeypatch.setattr(calibrate, 'CompactPINOv21', TinyModel)
    return manifest, packet_path, data, calls


def test_complete_paired_calibration_uses_shared_median_and_identical_e0(tmp_path, monkeypatch):
    manifest, packet, data, calls = calibration_fixture(tmp_path, monkeypatch)
    output = tmp_path/'calibration.json'
    report = calibrate.run_calibration(manifest, packet, output, mode='smoke', full_cell_rule='gl2', max_records=2)
    assert calls == [(str(manifest), 'smoke')]
    assert report['status'] == 'CALIBRATED_NOT_ACCEPTANCE'
    assert report['parameter_updates'] == 0
    assert report['training_manifest_sha256'] == data.manifest_sha256
    assert report['cohort_sha256'] == calibrate.sha256(packet)
    assert report['initial_weights_sha256'] is None
    assert report['initial_weights'] is None
    assert report['weights']['residual_loss_weight'] > 0 and report['weights']['constraint_loss_weight'] > 0
    assert report['calibration_rule']['valid_ratio_counts'] == {'R': 4, 'C': 4}
    assert all(row['shared_initial_response']['status'] == 'MATCHED' for row in report['records'])
    assert all(row['role'] == 'train' for row in report['records'])
    assert all(value['state_unchanged'] for value in report['provenance'].values())
    assert report['provenance']['bounded_cubic']['initial_state_sha256'] == report['provenance']['monotone_warp']['initial_state_sha256']
    for term, name in [('R', 'residual_loss_weight'), ('C', 'constraint_loss_weight')]:
        ratios = [arm['candidate_weight_ratios'][term] for row in report['records'] for arm in row['arms'].values()]
        assert report['weights'][name] == pytest.approx(float(np.median(ratios)))
    assert json.loads(output.read_text()) == report
    with pytest.raises(FileExistsError):
        calibrate.run_calibration(manifest, packet, output, mode='smoke')


def test_canonical_checkpoint_migration_is_bound_and_resets_only_output(tmp_path, monkeypatch):
    manifest, packet, data, _ = calibration_fixture(tmp_path, monkeypatch)
    torch.manual_seed(102)
    model = TinyModel()
    config = asdict(TrainConfig())
    path = tmp_path/'source.pt'
    torch.save(dict(schema=CHECKPOINT_SCHEMA, config=config, config_sha256=canonical_digest(config),
                    model=model.state_dict()), path)
    digest = calibrate.sha256(path)
    report = calibrate.run_calibration(manifest, packet, tmp_path/'from_checkpoint.json', mode='smoke',
        initial_weights=path, initial_weights_sha256=digest)
    assert report['initial_weights_sha256'] == digest
    assert report['initial_weights']['path'] == str(path.resolve())
    assert report['initial_weights']['sha256'] == digest
    assert report['initial_weights']['scope'] == 'geometry_and_hidden'
    for arm in report['provenance'].values():
        assert arm['initialization']['scope'] == 'geometry_and_hidden'
        assert arm['initialization']['reset_output_keys'] == ['response.local.2.weight', 'response.local.2.bias']
    with pytest.raises(ValueError, match='SHA256'):
        calibrate.run_calibration(manifest, packet, tmp_path/'bad_hash.json', mode='smoke',
            initial_weights=path, initial_weights_sha256='f'*64)
    torch.save({'private_model_weights': model.state_dict()}, path)
    with pytest.raises(ValueError, match='runner checkpoint'):
        calibrate.run_calibration(manifest, packet, tmp_path/'private.json', mode='smoke',
            initial_weights=path, initial_weights_sha256=calibrate.sha256(path))


def test_no_gradient_cannot_silently_generate_a_zero_physics_weight():
    records = [dict(arms={name: dict(probe=dict(gradient_norms=dict(MJ=1., R=0., C=2.)))
                         for name in calibrate.PROFILES})]
    with pytest.raises(ValueError, match='finite positive'):
        calibrate._shared_weights(records)
