"""Analytic software fixtures, explicitly not production or blind-test evidence."""
from __future__ import annotations

from types import SimpleNamespace
import hashlib
import numpy as np

from research.NeuralMoments.models.pcraft_ufno.candidate_projects.quadrann_reproduction.spin_adapter.v21_material import prepare_rle_material
from research.NeuralMoments.models.pcraft_ufno.candidate_projects.quadrann_reproduction.spin_adapter.v21_native import FiniteOperand
from .dataset import canonical_digest


def box_vertices(lower, upper):
    a,b = np.asarray(lower),np.asarray(upper)
    vertices = np.array([[a[0],a[1],a[2]],[b[0],a[1],a[2]],[b[0],b[1],a[2]],[a[0],b[1],a[2]],
                         [a[0],a[1],b[2]],[b[0],a[1],b[2]],[b[0],b[1],b[2]],[a[0],b[1],b[2]]],float)
    faces = np.array([[0,2,1],[0,3,2],[4,5,6],[4,6,7],[0,1,5],[0,5,4],
                      [1,2,6],[1,6,5],[2,3,7],[2,7,6],[3,0,4],[3,4,7]])
    return vertices,faces


def cube_material(side=16):
    columns = np.array([[x,y,0,side] for x in range(side) for y in range(side)],np.int32)
    return prepare_rle_material(columns,np.zeros(3),1./side,side)


def plane_operand(offset=.43, *, perturbation=None, operation='intersection', dimension=3):
    if dimension not in (3,8,10):
        raise ValueError('fixture derivative dimension must be 3, 8 or 10')
    vertices,faces = box_vertices([-2,-2,-2],[offset,2,2])
    moving = vertices[:,0] == offset
    motion = np.zeros((8,3,dimension))
    motion[moving,0,:3] = np.column_stack((np.ones(moving.sum()),vertices[moving,1]-.5,vertices[moving,2]-.5))
    if perturbation is not None:
        step = np.asarray(perturbation)
        if step.shape != (dimension,):
            raise ValueError('fixture perturbation has wrong chart dimension')
        vertices = vertices+np.einsum('vkd,d->vk',motion,step)
    # All six bounding faces belong to the finite operand. The four side/end
    # caps are explicitly tagged; no infinite-halfspace substitution is used.
    caps = np.ones(len(faces),bool)
    caps[[6,7]] = False
    return FiniteOperand(vertices,faces,caps,operation,motion)


def hard_cube_moments(offset):
    if not 0 < offset < 1:
        raise ValueError('analytic fixture must be away from true contact events')
    c = float(offset)
    moment = np.array([c,c*c/2,c/2,c/2,c**3/3,c/3,c/3,c/4,c*c/4,c*c/4])
    jac = np.array([[1,0,0],[c,0,0],[.5,1/12,0],[.5,0,1/12],[c*c,0,0],
                    [1/3,1/12,0],[1/3,0,1/12],[1/4,1/24,1/24],
                    [c/2,0,c/12],[c/2,c/12,0]])
    return moment,jac


def synthetic_dataset(side=16):
    """Two training queries; labels are independently integrated hard cuboids.

    Their 3-coordinate local face chart is an analytic fixture, not a claim of
    deployed R3 label qualification. The normal production loader is not bypassed
    by real-data commands; this fixture is reachable only via explicit smoke.
    """
    material = cube_material(side)
    fixed = .02*np.array([1,.25,.25,.25,.25**2,.25**2,.25**2,.25**2,.25**2,.25**2])
    full = np.array([1,.5,.5,.5,1/3,1/3,1/3,.25,.25,.25])
    rows = []
    for i,cut in enumerate((.43,.61)):
        value,jac = hard_cube_moments(cut)
        operand = plane_operand(cut)
        meta = dict(split='synthetic',reference_kind='analytic_fixture',geometry_id='synthetic-cube',
                    source_group='synthetic-cube-family',geometry_sha256=hashlib.sha256(b'unit-cube').hexdigest(),
                    material_identity=material.identity,teacher_version='analytic-cuboid-v1',
                    density=1.,editable_mass=1.,length_scale=1.,derivative_valid=True,
                    derivative_dimension=3,operation='intersection',surface_family='PLANE',
                    trajectory_stage='analytic_control',native_material_resolution=side,
                    query_content_sha256=canonical_digest([cut]),
                    operand_content_sha256=canonical_digest(operand.vertices.tolist()))
        case = dict(blocks=material,operand=operand,fixed_moments=fixed.copy(),native_moments=value+fixed,
                    native_full_editable_moments=full.copy(),native_jacobian=jac,
                    parameter_scales=np.ones(3),meta=meta)
        rows.append(dict(record_id=f'analytic-{i}',role='train',loader_eligible=True,
                         moment_weight=1.,jacobian_weight=1.,case=case,
                         case_sha256=canonical_digest(dict(meta=meta,moments=(value+fixed).tolist(),jacobian=jac.tolist()))))
    manifest = canonical_digest([row['case_sha256'] for row in rows])
    return SimpleNamespace(records=rows,train=rows,development=[],snapshot_id='synthetic-henmo-software-v1',
                           manifest_sha256=manifest,scope='synthetic_software_only',
                           source_admission={'status':'SYNTHETIC_SOFTWARE_ONLY','native_teacher':False})
