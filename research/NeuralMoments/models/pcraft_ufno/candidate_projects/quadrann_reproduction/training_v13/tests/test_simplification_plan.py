import json
import os
from pathlib import Path
import subprocess

import pytest
import torch
from torch import nn

from research.NeuralMoments.models.pcraft_ufno.candidate_projects.quadrann_reproduction.spin_adapter import ablation_initialization
from research.NeuralMoments.models.pcraft_ufno.candidate_projects.quadrann_reproduction.training_v13 import simplification_plan
from research.NeuralMoments.models.pcraft_ufno.candidate_projects.quadrann_reproduction.training_v13.tests import test_unified_corpus as corpus_fixtures


def git(root, *args):
    return subprocess.run(["git", "-C", str(root), *args], check=True, capture_output=True, text=True)


class TinyModel(nn.Module):
    def __init__(self, name):
        super().__init__()
        self.linear = nn.Linear(2, 2)
        self.name = name

    def architecture_contract(self):
        return {"name": self.name, "fixture": True}


@pytest.fixture
def plan_inputs(tmp_path, monkeypatch):
    root = tmp_path / "repository with spaces"
    root.mkdir()
    manifest = corpus_fixtures.UnifiedCorpusTests()._write_corpus(root)
    unusual = root / "manifest ' $(touch injected).json"
    manifest.rename(unusual)
    source = root / "research/NeuralMoments/models/pcraft_ufno"
    source.mkdir(parents=True)
    marker = source / "fixture.py"
    marker.write_text("# synthetic source\n")
    git(root, "init", "-q")
    git(root, "add", "--", ".")
    git(root, "-c", "user.name=Test", "-c", "user.email=test@example.invalid",
        "commit", "-qm", "synthetic fixture")
    builder = lambda name, **kwargs: TinyModel(name)
    monkeypatch.setattr(simplification_plan, "build_model", builder)
    monkeypatch.setattr(ablation_initialization, "build_model", builder)
    return root, unusual, marker


@pytest.mark.parametrize("stage,count", [("screen", 3), ("interaction", 4), ("scales", 3), ("all", 8)])
def test_plan_contains_unique_pinned_arms_without_starting_training(plan_inputs, stage, count):
    root, manifest, _ = plan_inputs
    output = root / "new plan ' $(touch injected)"
    plan = simplification_plan.generate_plan(manifest, output, stage=stage, repository_root=root)
    assert not plan["training_started"]
    assert len(plan["arms"]) == len({arm["config_name"] for arm in plan["arms"]}) == count
    assert plan["source_commit"] == git(root, "rev-parse", "HEAD").stdout.strip()
    assert plan["learning_rate"] == 3e-5 and plan["epochs"] == 3
    for arm in plan["arms"]:
        argv = arm["training_argv"]
        assert argv[argv.index("--manifest") + 1] == str(manifest)
        assert argv[argv.index("--initialization-reference") + 1] == "full_v1_candidate"
        assert arm["initialization"]["complete_target_coverage"]
        assert arm["architecture_manifest"]["name"] == arm["config_name"]
        assert not Path(arm["training_output"]).exists()
    assert sorted(p.name for p in output.iterdir()) == ["plan.json", "preflight.sh", "train.sh"]
    assert json.loads((output / "plan.json").read_text()) == plan
    for script in ("preflight.sh", "train.sh"):
        subprocess.run(["bash", "-n", str(output / script)], check=True)
    with pytest.raises(FileExistsError, match="new directory"):
        simplification_plan.generate_plan(manifest, output, repository_root=root)


def mock_python(root):
    wrapper = root / "mock python"
    wrapper.write_text(
        "#!/usr/bin/env python3\n"
        "import json,os,pathlib,sys\n"
        "if sys.argv[1] == '-c':\n"
        "    os.execv(sys.executable, [sys.executable, *sys.argv[1:]])\n"
        "with open(os.environ['TVCG_TEST_LOG'], 'a') as stream:\n"
        "    stream.write(json.dumps(sys.argv[1:]) + '\\n')\n"
    )
    wrapper.chmod(0o755)
    return wrapper


def test_generated_scripts_quote_paths_and_refuse_to_overwrite_runs(plan_inputs):
    root, manifest, _ = plan_inputs
    output = root / "plan ' $(touch injected)"
    simplification_plan.generate_plan(manifest, output, repository_root=root, geometry_id="train")
    log = root / "mock_calls.jsonl"
    env = dict(os.environ, TVCG_PYTHON=str(mock_python(root)), TVCG_TEST_LOG=str(log))
    for script in ("preflight.sh", "train.sh"):
        subprocess.run(["bash", str(output / script)], check=True, env=env, capture_output=True, text=True)
    calls = [json.loads(line) for line in log.read_text().splitlines()]
    assert len(calls) == 6
    assert all(call[call.index("--manifest") + 1] == str(manifest) for call in calls)
    assert all("--smoke-update" in call and "--geometry-id" in call for call in calls[:3])
    assert all("--initialization-reference" in call for call in calls[3:])
    assert not (root / "injected").exists()
    second = subprocess.run(["bash", str(output / "train.sh")], env=env, capture_output=True, text=True)
    assert second.returncode != 0 and "Refusing to overwrite" in second.stderr
    assert len(log.read_text().splitlines()) == 6


def test_script_keeps_failed_command_status_and_console_log(plan_inputs):
    root, manifest, _ = plan_inputs
    output = root / "plan"
    plan = simplification_plan.generate_plan(manifest, output, repository_root=root)
    wrapper = mock_python(root)
    with wrapper.open("a") as stream:
        stream.write("print('synthetic command failure')\nsys.exit(7)\n")
    log = root / "calls.jsonl"
    env = dict(os.environ, TVCG_PYTHON=str(wrapper), TVCG_TEST_LOG=str(log))
    result = subprocess.run(["bash", str(output / "train.sh")], env=env, capture_output=True, text=True)
    assert result.returncode == 7
    assert len(log.read_text().splitlines()) == 1
    console = Path(plan["arms"][0]["training_output"]) / "console.log"
    assert "synthetic command failure" in console.read_text()
    assert not Path(plan["arms"][1]["training_output"]).exists()


@pytest.mark.parametrize("changed", ["manifest", "commit", "source", "untracked_source"])
def test_script_pins_manifest_commit_and_tracked_source(plan_inputs, changed):
    root, manifest, marker = plan_inputs
    output = root / "plan"
    simplification_plan.generate_plan(manifest, output, repository_root=root)
    if changed == "manifest":
        manifest.write_text(manifest.read_text() + "\n")
    elif changed == "source":
        marker.write_text("# changed after planning\n")
    elif changed == "untracked_source":
        (marker.parent / "new_module.py").write_text("# not in the pinned commit\n")
    else:
        git(root, "-c", "user.name=Test", "-c", "user.email=test@example.invalid",
            "commit", "--allow-empty", "-qm", "new commit")
    log = root / "calls.jsonl"
    env = dict(os.environ, TVCG_PYTHON=str(mock_python(root)), TVCG_TEST_LOG=str(log))
    result = subprocess.run(["bash", str(output / "train.sh")], env=env, capture_output=True, text=True)
    assert result.returncode != 0
    assert not log.exists()
    assert not (output / "training").exists()


@pytest.mark.parametrize("untracked", [False, True])
def test_generation_rejects_uncommitted_source_before_writing_plan(plan_inputs, untracked):
    root, manifest, marker = plan_inputs
    if untracked:
        (marker.parent / "new_module.py").write_text("# untracked\n")
    else:
        marker.write_text("# modified\n")
    output = root / "not_created"
    with pytest.raises(ValueError, match="source"):
        simplification_plan.generate_plan(manifest, output, repository_root=root)
    assert not output.exists()


def test_cli_only_generates_selected_plan(plan_inputs, monkeypatch, capsys):
    root, manifest, _ = plan_inputs
    output = root / "cli plan"
    monkeypatch.chdir(root)
    monkeypatch.setattr("sys.argv", ["simplification_plan", "--manifest", str(manifest),
                        "--output", str(output), "--stage", "interaction", "--device", "cpu"])
    assert simplification_plan.main() == 0
    summary = json.loads(capsys.readouterr().out)
    assert summary["stage"] == "interaction" and len(summary["arms"]) == 4
    assert not summary["training_started"]
    assert not (output / "training").exists()


def test_v2_screen_waits_for_all_preflights_and_checks_artifacts(plan_inputs):
    root, manifest, _ = plan_inputs
    output = root / "three arm screen"
    plan = simplification_plan.generate_plan(manifest, output, stage="screen", repository_root=root)
    # Exercise generated shell verification independently of physical data admission.
    plan["stage"] = "v2_screen256"
    for phase, name in (("preflight", "preflight.sh"), ("training", "train.sh")):
        (output/name).write_text(simplification_plan._script(plan, phase))
    log = root/"invocations"
    wrapper = mock_python(root)
    env = dict(os.environ, TVCG_PYTHON=str(wrapper), TVCG_TEST_LOG=str(log))
    def execute(script):
        return subprocess.run(["bash", str(output/script)], env=env, capture_output=True, text=True)
    assert execute("train.sh").returncode != 0 and not log.exists()
    # Model three successful test-fixture executables. Each must leave all its evidence.
    wrapper.write_text(wrapper.read_text() +
        "if '--smoke-update' in sys.argv:\n"
        "    out=pathlib.Path(sys.argv[sys.argv.index('--output')+1])\n"
        "    for item in (out,out.with_suffix('.report.json'),out.parent/'memory_profile.json'):\n"
        "        item.write_text('fixture')\n")
    assert execute("preflight.sh").returncode == 0
    profile = Path(plan["arms"][-1]["preflight_output"])/"memory_profile.json"
    profile.write_text("changed")
    assert execute("train.sh").returncode != 0
    assert not any(Path(arm["training_output"]).exists() for arm in plan["arms"])
    profile.write_text("fixture")
    assert execute("train.sh").returncode == 0
