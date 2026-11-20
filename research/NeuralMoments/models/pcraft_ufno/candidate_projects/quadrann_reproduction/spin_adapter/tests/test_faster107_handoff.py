from __future__ import annotations

import base64
import hashlib
import json
from pathlib import Path
import shutil

import pytest

from research.NeuralMoments.models.pcraft_ufno.candidate_projects.quadrann_reproduction.spin_adapter import faster107_handoff as handoff


@pytest.fixture
def evidence(tmp_path, monkeypatch):
    root = tmp_path / "source"
    marker = tmp_path / "consumer_was_executed"
    index = b'{\r\n  "arbitrary_snapshot_key": "fixture", "unfollowed_npz_path": "/missing/source.npz"\r\n}\r\n'
    loader = (
        "from pathlib import Path\n"
        f"Path({str(marker)!r}).write_text('IMPORT SIDE EFFECT')\n"
        "def dangerous_default():\n"
        f"    Path({str(marker)!r}).write_text('DEFAULT SIDE EFFECT')\n"
        "class Faster107Corpus:\n"
        '    """Fixture corpus documentation; no real cohort is declared."""\n'
        "    def __init__(self, directory, include_partial=False, *, limit: int | None = None):\n"
        '        """Constructor documentation."""\n'
        "        raise RuntimeError('MUST NOT CONSTRUCT')\n"
        "    def read(self, key, /, *, request=dangerous_default()) -> dict:\n"
        '        """Read documentation."""\n'
        "        raise RuntimeError('MUST NOT CALL')\n"
    ).replace("\n", "\r\n").encode("utf-8")
    payloads = {
        handoff.EVIDENCE_FILES[0][0]: index,
        handoff.EVIDENCE_FILES[1][0]: '{"审计字段": [], "unassumed_structure": {"n": 2}}\n'.encode(),
        handoff.EVIDENCE_FILES[2][0]: loader,
        handoff.EVIDENCE_FILES[3][0]: b'{"fixture_report": true}\n',
    }
    for relative, payload in payloads.items():
        destination = root / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(payload)
    # The public CLI always uses the pinned production digest. Fixtures substitute
    # a known test digest here instead of exposing a production override switch.
    monkeypatch.setattr(handoff, "EXPECTED_INDEX_SHA256", hashlib.sha256(index).hexdigest())
    return root, payloads, marker


def test_handoff_copies_exact_bytes_and_inspects_consumer_without_execution(evidence, tmp_path):
    root, payloads, marker = evidence
    output = tmp_path / "inspection"
    before = {relative: ((root / relative).read_bytes(), (root / relative).stat().st_mtime_ns)
              for relative in payloads}
    result = handoff.collect_handoff(root, output)
    assert not marker.exists()
    assert result["actual_index_sha256"] == result["expected_index_sha256"]
    assert result["index_sha256_verified"] is True
    assert result["integration_ready"] is False
    assert result["optimizer_steps"] == result["model_forward_calls"] == result["teacher_calls"] == 0
    assert result["consumer_executed"] is False
    assert result["referenced_dataset_payloads_read"] is False
    assert json.loads((output / "handoff.json").read_text()) == result
    rows = {row["path"]: row for row in result["files"]}
    for relative, payload in payloads.items():
        assert (output / relative).read_bytes() == payload
        assert (root / relative).read_bytes() == before[relative][0]
        assert (root / relative).stat().st_mtime_ns == before[relative][1]
        assert rows[relative]["sha256"] == hashlib.sha256(payload).hexdigest()
        assert rows[relative]["size_bytes"] == len(payload)
        content = rows[relative]["content"]
        encoded = content.encode("utf-8") if rows[relative]["content_encoding"] == "utf-8" else base64.b64decode(content)
        assert encoded == payload
    assert rows[handoff.EVIDENCE_FILES[0][0]]["json_summary"] == {
        "root_type": "object", "top_level_keys": ["arbitrary_snapshot_key", "unfollowed_npz_path"]}
    assert rows[handoff.EVIDENCE_FILES[4][0]] == {
        "path": handoff.EVIDENCE_FILES[4][0], "required": False, "exists": False}
    interface = result["consumer_interface"]
    assert interface["definition_found"] is True
    assert interface["runtime_export_verified"] is False
    corpus = interface["definitions"][0]
    assert corpus["scope"] == "module"
    assert "no real cohort" in corpus["docstring"]
    methods = {entry["name"]: entry for entry in corpus["methods"]}
    assert methods["__init__"]["signature"] == "__init__(self, directory, include_partial=False, *, limit: int | None=None)"
    assert methods["read"]["signature"] == "read(self, key, /, *, request=dangerous_default()) -> dict"
    assert methods["read"]["docstring"] == "Read documentation."


def test_handoff_index_digest_mismatch_creates_no_output_or_parent(evidence, tmp_path, monkeypatch):
    root, _, _ = evidence
    output = tmp_path / "uncreated-parent" / "inspection"
    monkeypatch.setattr(handoff, "EXPECTED_INDEX_SHA256", "0" * 64)
    with pytest.raises(handoff.HandoffInspectionError, match="index SHA-256 mismatch"):
        handoff.collect_handoff(root, output)
    assert not output.parent.exists()


@pytest.mark.parametrize("relative_index", [0, 1, 2])
def test_handoff_missing_required_evidence_creates_no_output(evidence, tmp_path, relative_index):
    root, _, _ = evidence
    (root / handoff.EVIDENCE_FILES[relative_index][0]).unlink()
    output = tmp_path / "uncreated-parent" / "inspection"
    with pytest.raises(handoff.HandoffInspectionError, match="required evidence is missing"):
        handoff.collect_handoff(root, output)
    assert not output.parent.exists()


def test_handoff_existing_output_is_never_replaced(evidence, tmp_path):
    root, _, _ = evidence
    output = tmp_path / "existing"
    output.mkdir()
    sentinel = output / "preserve.bin"
    sentinel.write_bytes(b"existing user file")
    with pytest.raises(handoff.HandoffInspectionError, match="already exists"):
        handoff.collect_handoff(root, output)
    assert list(output.iterdir()) == [sentinel]
    assert sentinel.read_bytes() == b"existing user file"


@pytest.mark.parametrize("symlink_kind", ["file", "directory"])
def test_handoff_rejects_source_symlink_escape(evidence, tmp_path, symlink_kind):
    root, _, _ = evidence
    relative = Path(handoff.EVIDENCE_FILES[2][0])
    outside = tmp_path / "outside"
    if symlink_kind == "file":
        outside.write_bytes((root / relative).read_bytes())
        (root / relative).unlink()
        (root / relative).symlink_to(outside)
    else:
        source_parent = root / relative.parts[0]
        shutil.copytree(source_parent, outside)
        shutil.rmtree(source_parent)
        source_parent.symlink_to(outside, target_is_directory=True)
    output = tmp_path / "inspection"
    with pytest.raises(handoff.HandoffInspectionError, match="symlink"):
        handoff.collect_handoff(root, output)
    assert not output.exists()


def test_handoff_rejects_overlap_traversal_root_and_output_symlink(evidence, tmp_path):
    root, _, _ = evidence
    with pytest.raises(handoff.HandoffInspectionError, match="overlap"):
        handoff.collect_handoff(root, root / "inspection")
    with pytest.raises(handoff.HandoffInspectionError, match="traversal"):
        handoff.collect_handoff(root, tmp_path / "unused" / ".." / "inspection")
    with pytest.raises(handoff.HandoffInspectionError, match="non-root"):
        handoff.collect_handoff(Path("/"), tmp_path / "inspection")
    output_parent = tmp_path / "redirect"
    actual_parent = tmp_path / "actual"
    actual_parent.mkdir()
    output_parent.symlink_to(actual_parent, target_is_directory=True)
    with pytest.raises(handoff.HandoffInspectionError, match="symlinks"):
        handoff.collect_handoff(root, output_parent / "inspection")
    assert list(actual_parent.iterdir()) == []


def test_handoff_output_is_deterministic_and_optional_malformed_json_is_preserved(evidence, tmp_path):
    root, _, _ = evidence
    smoke = root / handoff.EVIDENCE_FILES[3][0]
    smoke.write_bytes(b'{"unfinished":\n')
    first, second = tmp_path / "first", tmp_path / "second"
    a = handoff.collect_handoff(root, first)
    b = handoff.collect_handoff(root, second)
    assert a == b
    assert (first / "handoff.json").read_bytes() == (second / "handoff.json").read_bytes()
    row = a["files"][3]
    assert row["exists"] and "json_parse_error" in row
    assert (first / handoff.EVIDENCE_FILES[3][0]).read_bytes() == smoke.read_bytes()


def test_handoff_non_utf8_python_is_preserved_and_statically_decoded(evidence, tmp_path):
    root, _, marker = evidence
    payload = b'# coding: latin-1\nclass Faster107Corpus:\n    """caf\xe9"""\n    pass\n'
    (root / handoff.EVIDENCE_FILES[2][0]).write_bytes(payload)
    output = tmp_path / "inspection"
    result = handoff.collect_handoff(root, output)
    loader = result["files"][2]
    assert loader["content_encoding"] == "base64"
    assert base64.b64decode(loader["content"]) == payload
    assert (output / loader["path"]).read_bytes() == payload
    assert result["consumer_interface"]["definitions"][0]["docstring"] == "café"
    assert not marker.exists()


@pytest.mark.parametrize("which", ["file", "total"])
def test_handoff_size_caps_fail_before_output(evidence, tmp_path, monkeypatch, which):
    root, payloads, _ = evidence
    if which == "file":
        monkeypatch.setattr(handoff, "MAX_FILE_BYTES", len(payloads[handoff.EVIDENCE_FILES[0][0]]) - 1)
    else:
        monkeypatch.setattr(handoff, "MAX_TOTAL_BYTES", sum(len(payloads[item[0]]) for item in handoff.EVIDENCE_FILES[:3]) - 1)
    output = tmp_path / "uncreated-parent" / "inspection"
    with pytest.raises(handoff.HandoffInspectionError, match="limit"):
        handoff.collect_handoff(root, output)
    assert not output.parent.exists()


def test_handoff_concurrent_output_creation_cannot_be_overwritten(evidence, tmp_path, monkeypatch):
    root, _, _ = evidence
    output = tmp_path / "inspection"
    publish = handoff._publish_new_directory

    def concurrent_publish(staged, target):
        target.mkdir()
        publish(staged, target)

    monkeypatch.setattr(handoff, "_publish_new_directory", concurrent_publish)
    with pytest.raises(handoff.HandoffInspectionError, match="already exists"):
        handoff.collect_handoff(root, output)
    assert output.is_dir() and list(output.iterdir()) == []
    assert list(tmp_path.glob(".inspection.staging-*")) == []


def test_handoff_reports_an_imported_class_as_unverified_without_importing(evidence, tmp_path):
    root, _, _ = evidence
    (root / handoff.EVIDENCE_FILES[2][0]).write_bytes(b'from nonexistent_module import Faster107Corpus\n')
    result = handoff.collect_handoff(root, tmp_path / "inspection")
    assert result["consumer_interface"]["definition_found"] is False
    assert result["consumer_interface"]["runtime_export_verified"] is False
    assert result["integration_ready"] is False


def test_handoff_cli_returns_failure_without_partial_output_and_has_no_digest_override(evidence, tmp_path, monkeypatch, capsys):
    root, _, _ = evidence
    output = tmp_path / "inspection"
    monkeypatch.setattr(handoff, "EXPECTED_INDEX_SHA256", "0" * 64)
    assert handoff.main(["--dataset-root", str(root), "--output", str(output)]) == 2
    assert "SHA-256 mismatch" in capsys.readouterr().err
    assert not output.exists()
    with pytest.raises(SystemExit) as exc:
        handoff.main(["--dataset-root", str(root), "--output", str(output), "--expected-index-sha256", "0" * 64])
    assert exc.value.code == 2
