import tempfile, unittest, ast
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
from research.NeuralMoments.r3_native_build_preflight_v1 import (
    REQUIRED_NATIVE_SYMBOLS, inspect_native_module, parse_cmake_cache, run_preflight
)

class R3NativeBuildPreflightTests(unittest.TestCase):
    def test_preflight_source_is_valid_python(self):\n        from pathlib import Path\n        source=Path("research/NeuralMoments/r3_native_build_preflight_v1.py").read_text()\n        ast.parse(source)\n\n    def test_required_native_symbols(self):
        fake=SimpleNamespace(**{name:object() for name in REQUIRED_NATIVE_SYMBOLS})
        fake.__name__="fake_native"
        self.assertTrue(inspect_native_module(fake)["native_api_available"])

    def test_missing_native_symbol_detected(self):
        fake=SimpleNamespace()
        r=inspect_native_module(fake)
        self.assertFalse(r["native_api_available"])
        self.assertIn(REQUIRED_NATIVE_SYMBOLS[0],r["missing_symbols"])

    def test_cmake_cache_options_parsed(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/"CMakeCache.txt"
            p.write_text("# cache\nMAKE_IT_STAND_BUILD_PYTHON:BOOL=ON\nMAKE_IT_STAND_BUILD_SPIN_IT_FASTER:BOOL=ON\nCMAKE_BUILD_TYPE:STRING=Release\n")
            r=parse_cmake_cache(p)
            self.assertEqual(r["MAKE_IT_STAND_BUILD_PYTHON"],"ON")
            self.assertEqual(r["CMAKE_BUILD_TYPE"],"Release")

    def test_auto_module_resolution_falls_back_to_top_level(self):
        fake_native=SimpleNamespace(**{name:object() for name in REQUIRED_NATIVE_SYMBOLS})
        fake_native.__file__=None
        fake_facade=SimpleNamespace(**{name:object() for name in (
          "SpinItFasterRequest","SpinSolverConfig","SpinEvaluatorConfig",
          "solve_spin_it_faster_fullrank_quadric")})
        def importer(name):
            if name=="make_it_stand._make_it_stand": raise ModuleNotFoundError(name)
            if name=="_make_it_stand": return fake_native
            if name=="make_it_stand.spin": return fake_facade
            raise ModuleNotFoundError(name)
        with patch("research.NeuralMoments.r3_native_build_preflight_v1.importlib.import_module",side_effect=importer):
            r=run_preflight()
        self.assertTrue(r["native_api_available"])
        self.assertEqual(r["checks"]["selected_module_name"],"_make_it_stand")

if __name__=="__main__": unittest.main()
