"""Certified moment-only replay of the original physical Cherchi Boolean.

The acceleration operates after the native Boolean. It transports its exact
rational vertices to the registered frame and uses the production MPFR/GMP
integral. Ambiguous winding falls back to the production orientation routine.
"""
from dataclasses import dataclass
from functools import lru_cache
import fcntl
import hashlib
import importlib.util
import os
from pathlib import Path
import subprocess
import sysconfig
import tempfile

import numpy as np

from research.NeuralMoments.models.pcraft_ufno.candidate_projects.quadrann_reproduction.spin_adapter.v21_native import physical_operand_vertices


@lru_cache(maxsize=1)
def load_transform():
    import pybind11
    source = Path(__file__).with_name('primal_transform.cpp')
    cxx = os.environ.get('CXX','c++')
    version = subprocess.check_output([cxx,'--version'])
    abi = sysconfig.get_config_var('SOABI')
    tag = hashlib.sha256(source.read_bytes()+version+abi.encode()+pybind11.__version__.encode()).hexdigest()
    folder = Path(tempfile.gettempdir())/f'tvcg-stratum-primal-{os.getuid()}'/tag
    folder.mkdir(mode=0o700,parents=True,exist_ok=True)
    name = '_tvcg_stratum_primal'
    library = folder/(name+sysconfig.get_config_var('EXT_SUFFIX'))
    with (folder/'build.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        if not library.exists():
            with tempfile.TemporaryDirectory(dir=folder) as build:
                tmp = Path(build)/library.name
                subprocess.run([cxx,'-std=c++17','-O3','-fno-fast-math','-shared','-fPIC',
                                '-I'+sysconfig.get_paths()['include'],'-I'+pybind11.get_include(),
                                str(source),'-o',str(tmp),'-lgmpxx','-lgmp'],check=True)
                tmp.replace(library)
    spec = importlib.util.spec_from_file_location(name,library)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@dataclass
class RetainedValue:
    moments: np.ndarray
    vertices: np.ndarray
    faces: np.ndarray
    packet: dict
    arithmetic: dict
    path: str


def retained_value(backend, operand):
    """Return the center/perturbed retained boundary and its true material M (including the fixed shell)."""
    from make_it_stand.verification import cherchi_inprocess as ci
    if backend.integral_profile != 'CERTIFIED_MPFR256_GMP_FALLBACK_V1':
        raise ValueError('requires registered certified material integral')
    req = backend.request
    physical = physical_operand_vertices(operand,req.target_axes,req.target_origin,backend.length,backend.normalization_origin)
    packet = backend._boolean_with_valid_source(physical,operand.faces,operand.operation)
    backend._validate_packet(packet)
    emitted = np.asarray(packet['vertices'],dtype=float)
    faces = np.asarray(packet['faces'],dtype=np.int64)
    original = packet['exact_vertices']
    transform = load_transform()
    try:
        vertices,tokens = transform.transform(original,emitted,faces,backend.normalization_origin,backend.length,
                                              req.target_origin,req.target_axes)
        method = 'coherent exact affine'
    except transform.ReferenceOrientation:
        rational = tuple(tuple(ci.exact_fraction(x) for x in row) for row in original)
        faces,_ = ci.normalize_closed_component_orientation(emitted,faces,rational_vertices=rational)
        vertices,target = backend._target_view(rational)
        tokens = [[str(x.numerator)+'/'+str(x.denominator) for x in row] for row in target]
        method = 'registered exact component normalization'
    arithmetic = backend.arithmetic.PreparedBoundary(tokens,faces)
    editable,proof = arithmetic.integrate_moments_certified(req.material_density)
    total = np.asarray(editable)+backend.fixed
    if proof.get('enclosure_certified') is not True or total.shape!=(10,) or not np.isfinite(total).all():
        raise ValueError('certified native moment integral unavailable')
    return RetainedValue(total,vertices,faces,packet,dict(proof),method)
