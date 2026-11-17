import numpy as np
import torch
import pytest

from training_v12.checkpoint import CheckpointManager
from training_v12.normalization import MomentNormalizer


def make_objects(tmp_path):
    model = torch.nn.Linear(10, 10)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=10
    )

    normalizer = MomentNormalizer(
        np.ones(10, dtype=np.float64),
        100,
    )

    manager = CheckpointManager(
        tmp_path,
        manifest_sha256="test-manifest",
        model_config={"preset": "test"},
    )

    return model, optimizer, scheduler, normalizer, manager


def test_checkpoint_roundtrip(tmp_path):
    model, optimizer, scheduler, norm, manager = make_objects(tmp_path)

    x = torch.randn(4, 10)
    y = model(x).sum()
    y.backward()
    optimizer.step()
    scheduler.step()

    expected = {
        k: v.clone()
        for k, v in model.state_dict().items()
    }

    path = manager.save(
        "latest",
        model,
        optimizer,
        scheduler,
        epoch=3,
        global_step=150,
        normalizer=norm,
    )

    with torch.no_grad():
        for p in model.parameters():
            p.add_(1.0)

    state = manager.load(
        path,
        model,
        optimizer,
        scheduler,
        restore_rng=False,
    )

    assert state["epoch"] == 3
    assert state["global_step"] == 150

    for k, v in model.state_dict().items():
        assert torch.equal(v, expected[k])


def test_checkpoint_rejects_manifest_mismatch(tmp_path):
    model, optimizer, scheduler, norm, manager = make_objects(tmp_path)

    path = manager.save(
        "latest", model, optimizer, scheduler,
        1, 10, norm,
    )

    wrong = CheckpointManager(
        tmp_path,
        manifest_sha256="different-manifest",
        model_config={"preset": "test"},
    )

    with pytest.raises(ValueError, match="manifest"):
        wrong.load(path, model)


def test_checkpoint_rejects_model_config_mismatch(tmp_path):
    model, optimizer, scheduler, norm, manager = make_objects(tmp_path)

    path = manager.save(
        "latest", model, optimizer, scheduler,
        1, 10, norm,
    )

    wrong = CheckpointManager(
        tmp_path,
        manifest_sha256="test-manifest",
        model_config={"preset": "different"},
    )

    with pytest.raises(ValueError, match="configuration"):
        wrong.load(path, model)
