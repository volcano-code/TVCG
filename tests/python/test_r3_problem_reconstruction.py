import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from research.NeuralMoments.r3_problem_reconstruction_v1 import SCHEMA, admission_report, load_case

def _write(root, name, data):
    p = root / name
    p.write_bytes(data)
    return hashlib.sha256(data).hexdigest()

class R3ProblemReconstructionTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.mesh_hash = _write(self.root, "mesh.obj", b"mesh")
        self.shell_hash = _write(self.root, "shell.bin", b"shell")
        self.domain_hash = _write(self.root, "domain.bin", b"domain")

    def tearDown(self):
        self.tmp.cleanup()

    def manifest(self):
        return {
            "schema": SCHEMA,
            "case_id": "M69-R3-test-001",
            "original_mesh": "mesh.obj",
            "original_mesh_sha256": self.mesh_hash,
            "fixed_shell": "shell.bin",
            "fixed_shell_sha256": self.shell_hash,
            "design_domain": "domain.bin",
            "design_domain_sha256": self.domain_hash,
            "axis": [0, 0, 1],
            "axis_origin": [0, 0, 0],
            "contact": [0, 0, 0],
            "pin_support": "explicit-source-record",
            "initial_material_policy": "explicit-source-record",
            "quadric_coefficients": [1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            "quadric_initialization_kind": "explicit-source-record",
            "quadric_source": "explicit-source-record",
        }

    def test_explicit_manifest_round_trip(self):
        p = self.root / "case.json"
        p.write_text(json.dumps(self.manifest()))
        case = load_case(p)
        case.validate_files(self.root)
        self.assertEqual(case.axis, (0.0, 0.0, 1.0))

    def test_physical_and_numerical_initial_states_are_separate(self):
        p = self.root / "case.json"
        p.write_text(json.dumps(self.manifest()))
        case = load_case(p)
        self.assertNotIn("quadric_coefficients", case.physical_initial_state())
        self.assertNotIn("original_mesh", case.numerical_initial_state())

    def test_hash_mismatch_fails_closed(self):
        record = self.manifest()
        record["original_mesh_sha256"] = "0" * 64
        p = self.root / "case.json"
        p.write_text(json.dumps(record))
        with self.assertRaisesRegex(ValueError, "SHA256 mismatch"):
            load_case(p).validate_files(self.root)

    def test_missing_semantics_are_not_inferred(self):
        record = self.manifest()
        record.pop("pin_support")
        p = self.root / "case.json"
        p.write_text(json.dumps(record))
        with self.assertRaisesRegex(ValueError, "incomplete R3 case"):
            load_case(p)

    def test_admission_remains_false_before_physical_audit(self):
        p = self.root / "case.json"
        p.write_text(json.dumps(self.manifest()))
        self.assertFalse(admission_report(load_case(p))["physical_admission"])

if __name__ == "__main__":
    unittest.main()
