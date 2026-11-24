"""Rebuild the side-by-side Python module from the pinned native build archive."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import tempfile

HERE = Path(__file__).resolve().parent
OLD = Path('/data/xvxinyao/moment-guided-spin-v1/code/make-it-balance-paper-sim/'
           'experiments/noodles_v8_20261017/native-side-build-job-03/output/build')
DEFAULT = Path('/tmp/tvcg-stratum-build')
VER = 'ALGEBRAIC_FINITE_GRAPH_EXTRUSION_R3_R8_V1'
INPUTS = ('bindings.cpp', 'stratum_api.h', 'stratum_geometry.inc', 'candidate03-stratum.patch',
          'upstream_inputs.json', 'build.py')


def sha(path):
    with Path(path).open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def check(path, expected):
    actual = sha(path)
    if actual != expected:
        raise ValueError(f'Build input digest mismatch at {path}: got {actual}, expected {expected}')
    return actual


def build(dest, old):
    old = old.resolve()
    dest = dest.resolve()
    if dest.is_relative_to(old.parents[3]) or dest == HERE or dest.is_relative_to(HERE):
        raise ValueError('Use an out-of-tree build directory outside the original archive')
    dest.mkdir(parents=True, exist_ok=True)
    recipe = {n:sha(HERE/n) for n in INPUTS}
    pinned = json.loads((HERE/'upstream_inputs.json').read_text())
    for name, digest in pinned.items():
        check(old/name, digest)
    receipt = json.loads((old/'../receipt.json').read_text())
    db = json.loads((old/'compile_commands.json').read_text())
    deps = {str(Path(key).resolve()): digest for key, digest in receipt['actual_dependency_sha256'].items()}
    recipes = {name: next(z for z in db if z['file'].endswith('native-side-source-03/'+name))
               for name in ('pybind_module.cpp', 'QuadricTriangulator.cpp')}
    for z in recipes.values():
        check(z['file'], receipt['isolated_candidate_inputs'][z['file']])
    linkfile = old/'CMakeFiles/_make_it_stand.dir/link.txt'
    original_link = shlex.split(linkfile.read_text())
    suffix = Path(receipt['module']).name.removeprefix('_make_it_stand')
    module_name = '_compact_stratum_v1' + suffix

    with tempfile.TemporaryDirectory(prefix='rank-chart-build-',dir=dest) as temp:
        stage = Path(temp)
        source = stage/'QuadricTriangulator.cpp'
        shutil.copyfile(recipes[source.name]['file'], source)
        subprocess.run(['patch','--batch','--fuzz=0','-p1','-i',str(HERE/'candidate03-stratum.patch')],
                       cwd=stage, check=True)
        # Some source-checkout headers advanced after the archived build. The
        # tracked TVCG checkout carries their original, receipt-identical bytes.
        # Shadow just the drifted includes; check every subsequently opened file.
        checkout = Path(recipes['pybind_module.cpp']['file']).parents[3]
        recovered = {}
        for name, digest in deps.items():
            archived = Path(name)
            if not archived.is_relative_to(checkout/'include') or sha(archived) == digest:
                continue
            rel = archived.relative_to(checkout)
            alternative = HERE.parents[3]/rel
            check(alternative, digest)
            local = stage/rel
            local.parent.mkdir(parents=True,exist_ok=True)
            shutil.copyfile(alternative,local)
            recovered[str(local)] = name
        headers = {str((HERE/n).resolve()) for n in INPUTS}
        headers.add(str(source))
        def compile_one(key, name, new_source):
            command = shlex.split(recipes[key]['command'])
            command[command.index('-o')+1] = str(stage/(name+'.o'))
            command[command.index('-c')+1] = str(new_source)
            dep = stage/(name+'.d')
            command[1:1] = ['-I'+str(HERE), '-I'+str(stage/'include'), '-ffile-prefix-map='+str(stage)+'=candidate03-stratum',
                            '-ffile-prefix-map='+str(HERE)+'=compact-stratum-source', '-MD', '-MF', str(dep)]
            subprocess.run(command,cwd=old,check=True)
            paths = shlex.split(dep.read_text().replace('\\\n',' ').split(':',1)[1])
            records = {}
            for word in paths:
                path = (old/word).resolve()
                if str(path) in headers:
                    continue
                name = recovered.get(str(path),str(path))
                if name not in deps:
                    raise ValueError(f'Native compile dependency missing from pinned receipt: {path}')
                records[name] = check(path, deps[name])
            return records
        jobs = [('pybind_module.cpp','bindings',HERE/'bindings.cpp'),
                ('QuadricTriangulator.cpp','geometry',source)]
        with ThreadPoolExecutor(max_workers=2) as workers:
            used = {}
            for result in workers.map(lambda t: compile_one(*t), jobs):
                used.update(result)
        # Export only this module initializer. Existing frozen pybind types and
        # symbols keep their original implementation when both modules are loaded.
        sym = stage/'exports'
        sym.write_text('{global: PyInit__compact_stratum_v1; local: *;};\n')
        link = [x for x in original_link if not x.endswith('.cpp.o')]
        link[link.index('-o')+1] = str(stage/module_name)
        where = link.index('-o')
        link[where:where] = [str(stage/'bindings.o'),str(stage/'geometry.o'),
                             '-Wl,--version-script='+str(sym)]
        subprocess.run(link,cwd=old,check=True)
        digest = sha(stage/module_name)
        dependency_digest = hashlib.sha256(json.dumps(used,sort_keys=True,separators=(',',':')).encode()).hexdigest()
        info = dict(schema='compact.stratum.native.build.1', source_map_version=VER, module_file=module_name,
                    module_sha256=digest, upstream_module_sha256=receipt['module_sha256'],
                    upstream_receipt_sha256=sha(old/'../receipt.json'), archive=str(old),
                    upstream_compiler_dependency_count=len(used), upstream_compiler_dependency_sha256=dependency_digest,
                    recovered_receipt_headers={n:deps[n] for n in sorted(recovered.values()) if n in used},
                    patched_triangulator_sha256=sha(source), recipe=recipe)
        if {n:sha(HERE/n) for n in INPUTS} != recipe:
            raise RuntimeError("Build recipe changed during compilation; rebuild from the final version")
        (stage/'build_receipt.json').write_text(json.dumps(info,sort_keys=True,indent=2)+'\n')
        os.replace(stage/module_name, dest/module_name)
        os.replace(stage/'build_receipt.json', dest/'build_receipt.json')
    print(f'{dest/module_name} {digest}',flush=True)
    return info


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--build-dir',type=Path,default=Path(os.environ.get('TVCG_STRATUM_BUILD',DEFAULT)))
    p.add_argument('--archive',type=Path,default=OLD)
    a=p.parse_args()
    build(a.build_dir,a.archive)
