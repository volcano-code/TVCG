"""Retain the exact linked object from a verified rebuild for the offline enumerator."""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import shlex
import shutil
import subprocess
from types import SimpleNamespace

HERE=Path(__file__).resolve().parent
NATIVE=HERE.parents[1]/'native_backend'
SYMBOL='_ZN12_GLOBAL__N_18classifyERKN4spin21QuadricCoefficients10EPN5Eigen22SelfAdjointEigenSolverINS4_6MatrixIdLi3ELi3ELi0ELi3ELi3EEEEE'

def run(dest):
    dest=Path(dest).resolve();dest.mkdir(parents=True,exist_ok=True)
    spec=importlib.util.spec_from_file_location('registered_stratum_build',NATIVE/'build.py')
    build=importlib.util.module_from_spec(spec);spec.loader.exec_module(build)
    def capture(args,**kw):
        if '-shared' in args:
            obj=next(Path(x) for x in args if Path(x).name=='geometry.o')
            shutil.copyfile(obj,dest/'original-geometry.o')
        return subprocess.run(args,**kw)
    build.subprocess=SimpleNamespace(run=capture)
    result=build.build(dest/'rebuild',build.OLD)
    ref=json.loads((NATIVE/'build_receipt.json').read_text())
    if result!=ref:raise ValueError('Verifier object came from a non-identical native module build')
    # Only symbol visibility changes in the offline object. The independent
    # binary calls the original compiled classifier and QR arithmetic.
    subprocess.run(['/usr/bin/objcopy','--globalize-symbol='+SYMBOL,str(dest/'original-geometry.o'),
                    str(dest/'test-geometry.o')],check=True)
    for name in ('original-geometry','test-geometry'):
        subprocess.run(['/usr/bin/objcopy','--dump-section','.text='+str(dest/(name+'.text')),
                        str(dest/(name+'.o')),str(dest/(name+'.checked.o'))],check=True)
    if build.sha(dest/'original-geometry.text')!=build.sha(dest/'test-geometry.text'):
        raise ValueError('Symbol export changed the native classifier machine code')
    original=build.OLD
    rec=json.loads((original/'compile_commands.json').read_text())
    z=next(z for z in rec if z['file'].endswith('native-side-source-03/pybind_module.cpp'))
    cmd=shlex.split(z['command'])
    # The same receipt-identical headers were required by the verified native build.
    restore=dest/'headers';root=Path(z['file']).parents[3]
    for p,digest in ref['recovered_receipt_headers'].items():
        rel=Path(p).relative_to(root/'include');source=NATIVE.parents[3]/'include'/rel
        build.check(source,digest); dst=restore/rel;dst.parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(source,dst)
    cmd.insert(1,'-I'+str(restore));cmd[cmd.index('-o')+1]=str(dest/'probe.o')
    cmd[cmd.index('-c')+1]=str(HERE/'probe.cpp')
    subprocess.run(cmd,cwd=original,check=True)
    name=ref['module_file'].replace('_compact_stratum_v1','_rank_admission_probe')
    link=shlex.split((original/'CMakeFiles/_make_it_stand.dir/link.txt').read_text())
    link=[a for a in link if not a.endswith('.cpp.o')]
    link[link.index('-o')+1]=str(dest/name)
    sym=dest/'symbols';sym.write_text('{global: PyInit__rank_admission_probe; local: *;};\n')
    where=link.index('-o');link[where:where]=[str(dest/'probe.o'),str(dest/'test-geometry.o'),'-Wl,--version-script='+str(sym)]
    subprocess.run(link,cwd=original,check=True)
    eigen=Path(z['file']).parents[3]/'libigl/external/eigen/Eigen/src'
    headers=('Eigenvalues/SelfAdjointEigenSolver.h','Eigenvalues/Tridiagonalization.h','Jacobi/Jacobi.h',
             'Core/MathFunctions.h','Core/MathFunctionsImpl.h')
    out=dict(toolchain_sha256={name:build.sha(name) for name in ('/usr/bin/c++','/usr/bin/ld','/usr/bin/objcopy')},
             native_geometry_text_sha256=build.sha(dest/'original-geometry.text'),
             eigen_header_sha256={name:build.sha(eigen/name) for name in headers},schema='tvcg.native-admission-probe.1',classified_compiler_object_sha256=build.sha(dest/'original-geometry.o'),
             verified_source_module_sha256=ref['module_sha256'],classifier_symbol=SYMBOL,
             probe_sha256=build.sha(dest/name),probe_cpp_sha256=build.sha(HERE/'probe.cpp'),
             probe_build_sha256=build.sha(HERE/'build_probe.py'))
    (dest/'probe_receipt.json').write_text(json.dumps(out,indent=2)+'\n')
    return out

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('destination',type=Path)
    print(run(parser.parse_args().destination))
