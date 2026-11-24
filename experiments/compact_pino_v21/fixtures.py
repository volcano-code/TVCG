"""Independent analytic cuboids for static and cut-response checks at native256."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from research.NeuralMoments.models.pcraft_ufno.candidate_projects.quadrann_reproduction.spin_adapter.v21_blocks import BlockMeasure
from research.NeuralMoments.models.pcraft_ufno.candidate_projects.quadrann_reproduction.spin_adapter.v21_material import prepare_material
from research.NeuralMoments.models.pcraft_ufno.candidate_projects.quadrann_reproduction.spin_adapter.v21_response import prepare_surface
from .diagnostics import moment_errors
from .replay import eval_partitioned, write


def box(lo, hi):
    a,b,c = lo
    u,v,w = hi
    vertices = np.array([[a,b,c],[u,b,c],[u,v,c],[a,v,c],[a,b,w],[u,b,w],[u,v,w],[a,v,w]])
    faces = np.array([[0,2,1],[0,3,2],[4,5,6],[4,6,7],[0,1,5],[0,5,4],
                      [1,2,6],[1,6,5],[2,3,7],[2,7,6],[3,0,4],[3,4,7]])
    return vertices, faces


def box_moments(lo, hi):
    lo, hi = np.array(lo), np.array(hi)
    d = hi-lo
    if (d <= 0).any():
        return np.zeros(10)
    m = np.prod(d)
    x,y,z = (lo+hi)*.5
    return m*np.r_[1,x,y,z,np.array([x*x,y*y,z*z])+d*d/12,y*z,x*z,x*y]


def setup(name):
    h = 1/256.
    if name == 'long_column':
        return [(1, [81*h,102*h,0], [82*h,103*h,1])], .53125
    if name == 'thin_wall':
        return [(1,[.161,.119,.121],[.161+.22*h,.191,.246])], .173
    if name == 'cavity':
        return [(1,[.121,.121,.121],[.209,.211,.221]),(-1,[.131,.142,.140],[.184,.189,.194])], .166
    if name == 'disconnected':
        return [(1,[.111,.112,.125],[.142,.138,.222]),(1,[.171,.151,.143],[.206,.204,.240])], .177
    raise ValueError(name)


def mesh_parts(parts):
    verts, inds, n = [], [], 0
    for sign, lo, hi in parts:
        v,f = box(lo,hi)
        verts.append(v); inds.append((f if sign>0 else f[:,::-1])+n)
        n += len(v)
    return np.vstack(verts), np.vstack(inds)


def manufactured_operand(height):
    v, f = box([-2.,-2.,height],[2.,2.,2.])
    velocity = np.zeros((len(v),3,3))
    velocity[:4,2,0] = 1.
    return v,f,velocity


def fixture(name, cache):
    parts, height = setup(name)
    v,f = mesh_parts(parts)
    material = prepare_material(v, f, np.zeros(3),1/256.)
    blocks = BlockMeasure.build(material, cache / name / material.identity)
    full = sum(sign*box_moments(a,b) for sign,a,b in parts)
    true_m, dy = np.zeros(10), np.zeros(10)
    for sign,a,b in parts:
        true_m += sign*box_moments([a[0],a[1],max(height,a[2])], b)
        if a[2] < height < b[2]:
            x,y = (np.array(a)+b)[:2]/2
            dx,dy_ = (np.array(b)-a)[:2]
            dy -= sign*dx*dy_*np.r_[1,x,y,height,x*x+dx*dx/12,y*y+dy_*dy_/12,
                                    height**2,y*height,x*height,x*y]
    sv,sf,velocity = manufactured_operand(height)
    surface = prepare_surface(sv,sf)
    fixed = box_moments([.63,.60,.57],[.635,.61,.58])
    answers, diag = eval_partitioned(blocks,surface,'intersection',fixed,velocity,1/256.)
    reverse, _ = eval_partitioned(blocks,surface,'difference',fixed,velocity,1/256.)
    length = np.ptp(v,axis=0).max()
    ref = true_m+fixed
    for k,y in answers.items():
        y['moment_error'] = moment_errors(y['M'],ref,full[0],length)
        y['center_z_abs'] = abs((y['M'][3]-fixed[3])/(y['M'][0]-fixed[0])-true_m[3]/true_m[0])
        y['height_derivative_error'] = moment_errors(np.array(y['J'])[:,0],dy,full[0]/length,length)
    return dict(case=name, nodes=len(blocks.nodes), static_error=moment_errors(blocks.moments.sum(0),full,full[0],length),
                true_M=ref.tolist(), true_dM_dheight=dy.tolist(), results=answers, counts=diag,
                fixed_moments=fixed.tolist(),
                opposite_max_abs=float(np.max(np.abs(np.array(answers['local']['M'])+reverse['local']['M']-full-2*fixed))),
                opposite_J_max_abs=float(np.max(np.abs(np.array(answers['local']['J'])+reverse['local']['J']))))


if __name__ == '__main__':
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--cache', type=Path, required=True)
    ap.add_argument('--output', type=Path, required=True)
    args=ap.parse_args()
    write(args.output,[fixture(n,args.cache) for n in ('long_column','thin_wall','cavity','disconnected')])
