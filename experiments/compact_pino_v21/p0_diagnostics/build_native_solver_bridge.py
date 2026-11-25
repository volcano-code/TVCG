"""Build only the existing repository numerical kernel; no geometry fallback.

Usage: python -m experiments.compact_pino_v21.p0_diagnostics.build_native_solver_bridge
           --eigen /absolute/eigen --output /absolute/build/libcpino_native.so
Dependencies are supplied explicitly. This script never installs or downloads.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shlex
import shutil
import subprocess
import tempfile

PROFILE = "TVCG_REPO_DUAL_MONOMIAL_RAW_V1_CABI_V1"


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def build(eigen, output, *, compiler="g++", include=(), link_libraries=()):
    root=Path(__file__).resolve().parents[3]
    eigen=Path(eigen).resolve()
    output=Path(output).resolve()
    if not (eigen / "Eigen/Core").is_file():
        raise ValueError("--eigen must contain Eigen/Core")
    executable=shutil.which(compiler)
    if not executable:
        raise RuntimeError("C++17 compiler unavailable")
    output.parent.mkdir(parents=True,exist_ok=True)
    sources=[Path(__file__).with_name("native_solver_bridge.cpp").resolve(),
             root / "src/spin/QuadricCoefficients.cpp",
             root / "src/spin/DualMonomialCoordinates.cpp",
             root / "src/spin/SpinMomentProblem.cpp",
             root / "src/spin/SpinDesignRequest.cpp",
             root / "src/manufacturing/SlicerValidation.cpp"]
    flags=["-std=c++17","-O0","-fPIC","-fvisibility=hidden",
           "-fvisibility-inlines-hidden","-ffunction-sections","-fdata-sections",
           "-DMAKE_IT_STAND_HAS_EXACT_GEOMETRY_CHECKS=0",
           "-I"+str(root / "include"),"-I"+str(root / "src"),"-I"+str(eigen),
           *("-I"+str(Path(path).resolve()) for path in include)]
    dependencies=set()
    commands=[]
    with tempfile.TemporaryDirectory(prefix="cpino_native_",dir=output.parent) as temp:
        objects=[]
        for index,source in enumerate(sources):
            target=Path(temp)/f"{index}.o"
            depfile=Path(temp)/f"{index}.d"
            command=[executable,*flags,"-MMD","-MF",str(depfile),"-c",str(source),"-o",str(target)]
            subprocess.run(command,check=True)
            commands.append(command)
            deptext=depfile.read_text().replace("\\\n", " ").split(":",1)[1]
            dependencies.update(Path(x).resolve() for x in shlex.split(deptext))
            objects.append(str(target))
        temporary=Path(temp)/output.name
        command=[executable,"-shared","-Wl,--gc-sections","-Wl,--no-undefined",*objects,
                 *("-l"+name for name in link_libraries),"-o",str(temporary)]
        subprocess.run(command,check=True)
        commands.append(command)
        temporary.replace(output)
    source_receipts={str(p.relative_to(root)):digest(p) for p in sorted(dependencies) if p.is_relative_to(root)}
    eigen_receipts={str(p.relative_to(eigen)):digest(p) for p in sorted(dependencies) if p.is_relative_to(eigen)}
    external_receipts={str(p):digest(p) for p in sorted(dependencies)
                       if not p.is_relative_to(root) and not p.is_relative_to(eigen)}
    receipt=dict(schema="cpino_native_numeric_build_v1",profile=PROFILE,
                 kernel_sha256=digest(output),kernel_path=str(output),
                 repository_root=str(root),source_sha256=source_receipts,
                 eigen_root=str(eigen),eigen_header_sha256=eigen_receipts,
                 external_header_sha256=external_receipts,
                 compiler=str(Path(executable).resolve()),
                 compiler_version=subprocess.check_output([executable,"--version"],text=True).splitlines()[0],
                 flags=flags,commands=commands,
                 claim_scope="repository_numerical_kernel_only_no_geometry_or_1756_equivalence")
    target=output.with_suffix(output.suffix+".build.json")
    target.write_text(json.dumps(receipt,indent=2,sort_keys=True)+"\n")
    return receipt


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--eigen",required=True,type=Path)
    parser.add_argument("--output",required=True,type=Path)
    parser.add_argument("--compiler",default="g++")
    parser.add_argument("--include",action="append",default=[],help="Additional real CGAL/Boost/MPFR headers")
    parser.add_argument("--link-library",action="append",default=[])
    args=parser.parse_args()
    receipt=build(args.eigen,args.output,compiler=args.compiler,include=args.include,link_libraries=args.link_library)
    print(json.dumps({k:receipt[k] for k in ("profile","kernel_path","kernel_sha256","claim_scope")},indent=2))


if __name__=="__main__":
    main()
