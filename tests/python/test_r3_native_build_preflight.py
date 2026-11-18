import tempfile, unittest
from pathlib import Path
from types import SimpleNamespace
from research.NeuralMoments.r3_native_build_preflight_v1 import (
    REQUIRED_NATIVE_SYMBOLS, inspect_native_module, parse_cmake_cache
)

class R3NativeBuildPreflightTests(unittest.TestCase):
    def test_required_native_symbols(self):
        fake=SimpleNamespace(**{name:object() for name in REQUIRED_NATIVE_SYMBOLS})
        fake.__name__="fake_native"
        r=inspect_native_module(fake)
        self.assertTrue(r["native_api_available"])
        self.assertEqual(r["missing_symbols"],[])

    def test_missing_native_symbol_detected(self):
        fake=SimpleNamespace(**{name:object() for name in REQUIRED_NATIVE_SYMBOLS[:-1]})
        r=inspect_native_module(fake)
        self.assertFalse(r["native_api_available"])
        self.assertIn(REQUIRED_NATIVE_SYMBOLS[-1],r["missing_symbols"])

    def test_cmake_cache_options_parsed(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/"CMakeCache.txt"
            p.write_text("# cache\n" + "MAKE_IT_STAND_BUILD_PYTHON:BOOL=ON\n" + "MAKE_IT_STAND_BUILD_SPIN_IT_FASTER:BOOL=ON\n" + "CMAKE_BUILD_TYPE:STRING=Release\n")
            r=parse_cmake_cache(p)
            self.assertEqual(r["MAKE_IT_STAND_BUILD_PYTHON"],"ON")
            self.assertEqual(r["CMAKE_BUILD_TYPE"],"Release")

if __name__=="__main__": unittest.main()
