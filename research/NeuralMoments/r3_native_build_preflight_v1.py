"""R3.1.1 native build preflight: verify the installed pybind extension and build provenance.

This checks availability and build configuration only. It does not execute a physical
solver case and never labels a native module as an exact M69 replay.
"""
from __future__ import annotations
import argparse, hashlib, importlib, json, os, sys
from pathlib import Path
from typing import Any

SCHEMA = "tvcg.r3.native-build-preflight.v1"
REQUIRED_NATIVE_SYMBOLS = (
    "solve_spin_it_faster_fullrank_quadric",
    "SpinItFasterRequest",
    "SpinSolverConfig",
    "SpinEvaluatorConfig",
)

def sha256(path: Path) -> str:
    h=hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda:f.read(1024*1024),b""): h.update(chunk)
    return h.hexdigest()

def parse_cmake_cache(path: str | Path) -> dict[str,str]:
    p=Path(path)
    if not p.is_file(): raise FileNotFoundError(p)
    result={}
    for line in p.read_text(errors="replace").splitlines():
        if not line or line.startswith(("#","//")) or "=" not in line or ":" not in line: continue
        left,value=line.split("=",1); key=left.split(":",1)[0]
        result[key]=value
    return result

def inspect_native_module(module: Any) -> dict:
    missing=[name for name in REQUIRED_NATIVE_SYMBOLS if not hasattr(module,name)]
    module_file=getattr(module,"__file__",None)
    return {
      "module_name":getattr(module,"__name__",type(module).__name__),
      "module_file":str(module_file) if module_file else None,
      "module_file_sha256":sha256(Path(module_file)) if module_file and Path(module_file).is_file() else None,
      "required_symbols":list(REQUIRED_NATIVE_SYMBOLS),
      "missing_symbols":missing,
      "native_api_available":not missing,
    }

def run_preflight(*,build_dir=None,expected_commit=None,module_name="make_it_stand._make_it_stand"):
    try:
        module=importlib.import_module(module_name)
    except Exception as exc:
        return {"schema":SCHEMA,"ok":False,"native_api_available":False,
          "module_name":module_name,"import_error":f"{type(exc).__name__}: {exc}",
          "build_configuration":{},"checks":{"extension_import":False}}
    info=inspect_native_module(module)
    config={}
    cache_path=None
    if build_dir:
        candidate=Path(build_dir)/"CMakeCache.txt"
        cache_path=str(candidate)
        if candidate.is_file():
            cache=parse_cmake_cache(candidate)
            config={k:cache.get(k) for k in (
              "MAKE_IT_STAND_BUILD_PYTHON","MAKE_IT_STAND_BUILD_SPIN_IT_FASTER",
              "CMAKE_BUILD_TYPE","CMAKE_CXX_COMPILER","Python_EXECUTABLE",
              "pybind11_DIR") if k in cache}
        else:
            config={"cmake_cache_missing":True}
    checks={
      "extension_import":True,
      "required_native_symbols":info["native_api_available"],
      "python_build_enabled":config.get("MAKE_IT_STAND_BUILD_PYTHON")=="ON" if config else None,
      "spin_it_faster_enabled":config.get("MAKE_IT_STAND_BUILD_SPIN_IT_FASTER")=="ON" if config else None,
      "expected_commit_match":None,
    }
    if expected_commit is not None:
        actual=os.environ.get("TVCG_SOURCE_COMMIT")
        checks["expected_commit_match"]=(actual==expected_commit) if actual else False
    hard=[checks["extension_import"],checks["required_native_symbols"]]
    if config and not config.get("cmake_cache_missing"):
        hard.extend([checks["python_build_enabled"],checks["spin_it_faster_enabled"]])
    if expected_commit is not None: hard.append(checks["expected_commit_match"])
    return {"schema":SCHEMA,"ok":all(x is True for x in hard),
      "native_api_available":info["native_api_available"],"module":info,
      "build_dir":str(build_dir) if build_dir else None,"cmake_cache":cache_path,
      "build_configuration":config,"checks":checks,
      "scope":"native extension preflight only; no geometry admission or solver replay"}

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--build-dir")
    p.add_argument("--expected-commit")
    p.add_argument("--module",default="make_it_stand._make_it_stand")
    p.add_argument("--output",required=True)
    a=p.parse_args()
    report=run_preflight(build_dir=a.build_dir,expected_commit=a.expected_commit,module_name=a.module)
    out=Path(a.output); out.parent.mkdir(parents=True,exist_ok=True)
    out.write_text(json.dumps(report,indent=2,sort_keys=True)+"\n")
    print(json.dumps(report,indent=2))
    raise SystemExit(0 if report["ok"] else 2)

if __name__=="__main__": main()
