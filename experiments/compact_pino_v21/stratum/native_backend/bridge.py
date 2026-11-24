"""Load the verified versioned source beside the frozen Cherchi/physics runtime."""
from copy import deepcopy
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
VERSION = 'ALGEBRAIC_FINITE_GRAPH_EXTRUSION_R3_R8_V1'


def _sha(path):
    with Path(path).open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


class NativeSource:
    def __init__(self, module, identity):
        self.source_value = module.source_value
        self.source_motion = module.source_motion
        self.full_rank_control = module.full_rank_control
        self._identity = identity

    @property
    def backend_identity(self):
        return deepcopy(self._identity)


def load_native(build_dir=None):
    root = Path(build_dir or os.environ.get('TVCG_STRATUM_BUILD','/tmp/tvcg-stratum-build')).resolve()
    record = json.loads((root/'build_receipt.json').read_text())
    reference = json.loads((HERE/'build_receipt.json').read_text())
    pins = json.loads((HERE/'upstream_inputs.json').read_text())
    expected_recipe = ('bindings.cpp','stratum_api.h','stratum_geometry.inc','candidate03-stratum.patch',
                       'upstream_inputs.json','build.py')
    if (record['module_sha256'] != reference['module_sha256'] or record['source_map_version'] != VERSION or
            record['module_file'] != reference['module_file'] or
            record['upstream_compiler_dependency_sha256'] != reference['upstream_compiler_dependency_sha256'] or
            record['upstream_module_sha256'] != reference['upstream_module_sha256'] or
            record['patched_triangulator_sha256'] != reference['patched_triangulator_sha256'] or
            record['upstream_receipt_sha256'] != pins['../receipt.json']
            or {n:_sha(HERE/n) for n in expected_recipe} != record['recipe']):
        raise ValueError('Installed native rank-chart module does not match its pinned build recipe')
    path = root/record['module_file']
    digest = _sha(path)
    if digest != record['module_sha256']:
        raise ValueError('Rank-chart module differs from its build receipt')
    name = '_compact_stratum_v1'
    module = sys.modules.get(name)
    if module is None:
        spec = importlib.util.spec_from_file_location(name,path)
        if spec is None or spec.loader is None:
            raise ImportError(f'Cannot load rank-chart native library at {path}')
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        module.__compact_stratum_loaded_sha256__ = digest
        sys.modules[name] = module
    if (Path(module.__file__).resolve() != path or module.source_map_version != VERSION or
            getattr(module,'__compact_stratum_loaded_sha256__',None) != digest):
        raise RuntimeError('A different native rank-chart module is already loaded in this process')
    return NativeSource(module,dict(source_map_version=VERSION,module_sha256=digest,module_file=path.name,
        upstream_module_sha256=record['upstream_module_sha256'],upstream_receipt_sha256=record['upstream_receipt_sha256'],
        patched_triangulator_sha256=record['patched_triangulator_sha256'],recipe=record['recipe'],
        upstream_compiler_dependency_sha256=record['upstream_compiler_dependency_sha256']))
