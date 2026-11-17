"""Differentiable task69 moment equations; no claim of material feasibility.

This module does not alter TaskSpec or its single-density material policy.
Variable density is an explicitly separate experimental relaxation. A moment
loss cannot establish connectivity, manufacturability or hard-cut feasibility.
"""
import math
import torch
from task69 import TaskSpec


def second_matrix(m):
    return torch.stack((torch.stack((m[...,4],m[...,9],m[...,8]),-1),
                        torch.stack((m[...,9],m[...,5],m[...,7]),-1),
                        torch.stack((m[...,8],m[...,7],m[...,6]),-1)),-2)


def physical(m,task,reference_polar=None):
    axis=m.new_tensor(task.axis);origin=m.new_tensor(task.axis_origin)
    contact=m.new_tensor(task.contact); eye=torch.eye(3,dtype=m.dtype,device=m.device)
    mass=m[...,0]; first=m[...,1:4]; center=first/mass[...,None]
    S=second_matrix(m)
    cov=S-mass[...,None,None]*center[..., :,None]*center[...,None,:]
    inertia=cov.diagonal(dim1=-2,dim2=-1).sum(-1)[...,None,None]*eye-cov
    shifted=S-origin[:,None]*first[...,None,:]-first[..., :,None]*origin[None,:]+mass[...,None,None]*torch.outer(origin,origin)
    polar=shifted.diagonal(dim1=-2,dim2=-1).sum(-1)-torch.einsum('i,...ij,j->...',axis,shifted,axis)
    reference_polar=polar if reference_polar is None else reference_polar
    delta=center-origin
    transverse=delta-(delta*axis).sum(-1)[...,None]*axis
    Ia=inertia@axis; axial=(Ia*axis).sum(-1)
    seed=eye[int(axis.abs().argmin())]
    u=torch.linalg.cross(axis,seed);u=u/torch.linalg.vector_norm(u)
    v=torch.linalg.cross(axis,u); plane=torch.stack((u,v),-1)
    transverse_max=torch.linalg.eigvalsh(plane.T@inertia@plane)[...,-1]
    return {'mass':mass,'polar':polar,'centroid':center,
            'transverse_centroid':torch.linalg.vector_norm(transverse,dim=-1),
            'centroid_height':((center-contact)*axis).sum(-1),
            'alignment_error':torch.linalg.vector_norm(Ia-axial[...,None]*axis,dim=-1)/reference_polar,
            'principal_gap':(axial-transverse_max)/reference_polar,
            'covariance_min_eigenvalue':torch.linalg.eigvalsh(cov)[...,0]}


def rotation_objective(moments, original, task: TaskSpec, length: float):
    """Return dimensionless per-query losses and frozen task69 constraints.

    `original` must be the frozen full unmodified material moments [10], not
    the current density's full moments. It must not require gradients. The
    caller chooses term weights; no optimizer or learned weights live here.
    Training polar loss uses task.training_retention_target, while reported
    physical constraints retain task.polar_retention and frozen tolerances.
    """
    if not isinstance(task,TaskSpec) or not math.isfinite(length) or length<=0:
        raise ValueError('frozen TaskSpec and positive finite length required')
    if (moments.ndim!=2 or moments.shape[-1]!=10 or moments.shape[0]==0
            or moments.dtype not in (torch.float32,torch.float64)):
        raise ValueError('moments must be nonempty float32/64 [Q,10]')
    if original.shape!=(10,) or original.requires_grad:
        raise ValueError('original must be frozen ten moments')
    if original.device!=moments.device or original.dtype!=moments.dtype:
        raise ValueError('original must share moment device and dtype')
    if not torch.isfinite(moments).all() or not torch.isfinite(original).all():
        raise ValueError('finite moments required')
    if (moments[:,0]<=0).any() or original[0]<=0:
        raise ValueError('positive current and original mass required')
    ref=physical(original,task)
    if ref['polar']<=0 or not torch.isfinite(ref['polar']):
        raise ValueError('original fixed-axis inertia must be positive')
    p=physical(moments,task,ref['polar'])
    mass_fraction=p['mass']/ref['mass']; retention=p['polar']/ref['polar']
    constraints={
        'positive_mass':mass_fraction,
        'no_added_mass':1+task.mass_tolerance-mass_fraction,
        'minimum_removal':1-mass_fraction-task.min_removal_fraction+task.mass_tolerance,
        'polar_retention':retention-task.polar_retention+task.inertia_tolerance,
        'no_added_polar_inertia':1+task.inertia_tolerance-retention,
        'axis_alignment_com':task.com_tolerance-p['transverse_centroid']/length,
        'axis_alignment_inertia':task.alignment_tolerance-p['alignment_error'],
        'principal_dominance':p['principal_gap']-task.principal_gap_min,
        'centroid_height':task.max_centroid_rise-(p['centroid_height']-ref['centroid_height'])/length,
        'physical_second_moments':p['covariance_min_eigenvalue']/(ref['mass']*length**2)+task.moment_psd_tolerance,
    }
    aliases={'com':'axis_alignment_com','alignment':'axis_alignment_inertia',
             'principal':'principal_dominance','height':'centroid_height',
             'minimum_removal':'minimum_removal','no_added_mass':'no_added_mass',
             'no_added_polar':'no_added_polar_inertia','second_moments':'physical_second_moments'}
    losses={key:torch.relu(-constraints[value]).square() for key,value in aliases.items()}
    losses['polar']=torch.relu(task.training_retention_target-retention-task.inertia_tolerance).square()
    losses['mass']=mass_fraction
    return {'losses':losses,'constraints':constraints,
            'metrics':{**p,'mass_fraction':mass_fraction,'polar_retention':retention},
            'task_hash':task.task_hash,'material_policy':'bounded_variable_density_relaxation.v1',
            'connectivity_checked':False,'training_polar_margin':task.training_retention_target-task.polar_retention}
