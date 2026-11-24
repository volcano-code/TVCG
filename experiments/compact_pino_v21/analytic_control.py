"""Analytic positive quadric versus the deployed finite operand on identical voxel nodes.

Both controls use the same compact cubic of signed Euclidean distance. For the
quadric, a strictly positive closest-point Lagrangian Hessian certifies the
global nearest solution. An unresolved medial-axis case withholds the result.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import time

import numpy as np

from research.NeuralMoments.models.pcraft_ufno.candidate_projects.quadrann_reproduction.spin_adapter.v21_blocks import (
    BlockMeasure, basis_at,
)
from .diagnostics import accepted_directions, jacobian_errors, moment_errors
from .replay import read, write


def _form(q):
    q = np.asarray(q, float)
    if q.shape != (10,) or not np.isfinite(q).all() or not np.any(q):
        raise ValueError('invalid coefficients')
    return np.array([[q[4], q[9], q[8]], [q[9], q[5], q[7]], [q[8], q[7], q[6]]])


def point_response(points, q, h):
    q = np.asarray(q, float)
    a = _form(q)
    eig, rot = np.linalg.eigh(a)
    x, lin = points @ rot, q[1:4] @ rot
    f = q[0] + x @ lin + (x*x) @ eig
    grad = 2*x*eig+lin
    p = np.asarray(f >= 0, dtype=float)
    j = np.zeros((len(x), 10))
    # The polynomial cannot change sign in the closed radius-h ball of far points.
    near = np.abs(f) <= h*np.linalg.norm(grad,axis=1) + h*h*np.abs(eig).max()
    if not near.any():
        return p, j
    xx, lam = x[near], np.zeros((int(near.sum()), 1))
    lower = np.max(-.5/eig[eig>0]) if np.any(eig>0) else -np.inf
    upper = np.min(-.5/eig[eig<0]) if np.any(eig<0) else np.inf
    for _ in range(30):
        den = 1 + 2*lam*eig
        y = (xx-lam*lin) / den
        fy = q[0] + y @ lin + (y*y) @ eig
        g = 2*y*eig+lin
        residual_bound = 128*np.finfo(float).eps*(abs(q[0]) + abs(y) @ abs(lin) + (y*y) @ abs(eig))
        if np.all(abs(fy) <= residual_bound):
            break
        slope = ((g*g)/den).sum(axis=1)
        if np.any(slope == 0):
            raise ValueError('undefined nearest-quadric direction')
        new = lam[:,0] + fy/slope
        new = np.where(new <= lower, .5*(lam[:,0]+lower), new)
        new = np.where(new >= upper, .5*(lam[:,0]+upper), new)
        lam = new[:,None]
    else:
        raise ValueError('no certified regular nearest point for analytic control')
    if (den <= 0).any():
        raise ValueError('closest-point Hessian not positive definite')
    d = np.linalg.norm(xx-y,axis=1)*np.sign(f[near])
    select = np.abs(d) < h
    s = np.clip(d/h,-1,1)
    t = (1+s)*.5
    p[near] = t*t*(3-2*t)
    if select.any():
        yy, normal_length = y[select] @ rot.T, np.linalg.norm(g[select],axis=1)
        if np.any(normal_length==0):
            raise ValueError('singular analytic level-set gradient')
        basis = basis_at(yy)
        basis[:,7:] *= 2
        j[np.flatnonzero(near)[select]] = basis * (.75*(1-s[select]**2)/(h*normal_length))[:,None]
    return p, j


def evaluate(blocks, q, fixed):
    h = blocks.pitch
    a = _form(q)
    c = blocks.centers
    radius = blocks.radius + h
    f = q[0] + c @ q[1:4] + ((c@a)*c).sum(1)
    bound = radius*np.linalg.norm(2*c@a+q[1:4],axis=1) + radius**2*np.linalg.norm(a,ord=2)
    tol = 128*np.finfo(float).eps * (abs(q).sum()*(1+abs(c).max())**2)
    value = fixed + blocks.moments[f > bound+tol].sum(0)
    jac = np.zeros((10,10))
    for nodes, w in blocks.chunks(abs(f) <= bound+tol):
        p, dp = point_response(nodes, q, h)
        wb = basis_at(nodes)*w[:,None]
        value += wb.T @ p
        jac += wb.T @ dp
    return value, jac


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--native-root', type=Path, required=True)
    ap.add_argument('--indexes', type=Path, required=True)
    ap.add_argument('--output', type=Path, required=True)
    ap.add_argument('--trajectory', type=Path)
    ap.add_argument('rows', nargs='+')
    args = ap.parse_args()
    dirs = accepted_directions(read(args.trajectory)['solver']['trace']) if args.trajectory else {}
    files = sorted({x for name in args.rows for x in (args.native_root/'rows').glob(name+'.json')})
    for path in files:
        row = read(path)
        if not row['moment_valid'] or not row['full_j_valid'] or row.get('same_operand_as'):
            continue
        name, gid = row['row_id'], row['geometry_id']
        meta = read(args.indexes/gid/'identity.json')
        geom = BlockMeasure.open(args.indexes/gid/'rle', identity=meta['controls']['rle']['identity'], block_voxels=8)
        t0 = time.perf_counter()
        q = np.array(row['query'])
        try:
            m, j = evaluate(geom, q, np.array(meta['fixed_moments']))
            mass, L = meta['reference_editable_mesh'][0], meta['length']
            v = dirs.get(int(name.split('_')[-1])) if gid == 'noodles' else None
            result = dict(M=m.tolist(), J=j.tolist(), moment_error=moment_errors(m,row['M'],mass,L),
                          jacobian_error=jacobian_errors(j,row['J_raw'],mass,L,np.linalg.norm(q),direction=v))
        except ValueError as exc:
            result = dict(defined=False, diagnostic=str(exc))
        write(args.output/(name+'.json'), dict(row_id=name, geometry=gid, result=result, elapsed_s=time.perf_counter()-t0))
        print(name, result.get('diagnostic', 'defined'), flush=True)


if __name__ == '__main__':
    main()
