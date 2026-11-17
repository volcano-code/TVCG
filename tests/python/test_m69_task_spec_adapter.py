import hashlib
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

from research.NeuralMoments.m69_task_spec_adapter_v1 import (
    SCHEMA_VERSION, extract_task_spec, load_task69_module,
)


FAKE_TASK_SOURCE = """
class TaskSpec:
    def __init__(self, density=1000.0, native_resolution=64, axis=(0.0,0.0,1.0),
                 contact=(0.0,0.0,0.0), tolerance=1e-8):
        self.density = float(density)
        self.native_resolution = int(native_resolution)
        self.axis = tuple(axis)
        self.contact = tuple(contact)
        self.tolerance = float(tolerance)
        self.task_hash = "fake-task-hash-v1"
    def to_dict(self):
        return {"density":self.density,"native_resolution":self.native_resolution,
                "axis":list(self.axis),"contact":list(self.contact),"tolerance":self.tolerance}
    @classmethod
    def from_dict(cls, value):
        return cls(**value)
"""


class M69TaskSpecAdapterTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.source = self.root / "task69.py"
        self.source.write_text(FAKE_TASK_SOURCE)
        self.sha = hashlib.sha256(self.source.read_bytes()).hexdigest()

    def tearDown(self):
        self.tmp.cleanup()

    def test_exact_source_hash_is_required(self):
        with self.assertRaisesRegex(ValueError, "SHA256 mismatch"):
            load_task69_module(self.source, expected_sha256="0" * 64)

    def test_extracts_all_fields_without_semantic_guessing(self):
        result = extract_task_spec(self.source, expected_sha256=self.sha)
        self.assertEqual(result["schema_version"], SCHEMA_VERSION)
        self.assertEqual(result["task"]["field_names"],
                         ["axis", "contact", "density", "native_resolution", "tolerance"])
        self.assertEqual(result["task"]["density"], 1000.0)
        self.assertEqual(result["task"]["native_resolution"], 64)
        self.assertIsNone(result["semantic_mapping"]["axis"])
        self.assertFalse(result["claims"]["physical_admission"])

    def test_accepts_complete_explicit_payload_and_canonicalizes(self):
        payload = {"density": 850.0, "native_resolution": 128,
                   "axis": [1.0, 0.0, 0.0], "contact": [0.0, 0.0, 0.0],
                   "tolerance": 1e-7}
        result = extract_task_spec(self.source, payload, expected_sha256=self.sha)
        self.assertEqual(result["task"]["density"], 850.0)
        self.assertEqual(result["task"]["native_resolution"], 128)
        self.assertEqual(result["task"]["payload"]["axis"], [1.0, 0.0, 0.0])

    def test_missing_task_field_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "incomplete TaskSpec"):
            extract_task_spec(self.source, {"density": 1.0}, expected_sha256=self.sha)

    def test_nonfinite_payload_is_rejected(self):
        payload = {"density": 850.0, "native_resolution": 128,
                   "axis": [1.0, 0.0, 0.0], "contact": [0.0, 0.0, 0.0],
                   "tolerance": float("nan")}
        with self.assertRaisesRegex(ValueError, "nonfinite"):
            extract_task_spec(self.source, payload, expected_sha256=self.sha)

    def test_density_must_be_positive(self):
        payload = {"density": -1.0, "native_resolution": 128,
                   "axis": [1.0, 0.0, 0.0], "contact": [0.0, 0.0, 0.0],
                   "tolerance": 1e-7}
        with self.assertRaisesRegex(ValueError, "density"):
            extract_task_spec(self.source, payload, expected_sha256=self.sha)


if __name__ == "__main__":
    unittest.main()
