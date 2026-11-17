"""Phase 2.1: analytic Pure-Quadric material operator contract.

The analytic reference uses a box clipped by the regular half-space
q(x)=c+b_x*x >= 0, with b_x<0 (retained region x<=a=-c/b_x).
The ten coefficient basis follows M80 QUERY_ORDER:
[1,x,y,z,x²,y²,z²,2yz,2xz,2xy].
This is an exact mathematical fixture, not an M80 model result or real-geometry
generalization claim.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Literal

import numpy as np

MOMENT_EXPONENTS = (
    (0,0,0),(1,0,0),(0,1,0),(0,0,1),(2,0,0),
    (0,2,0),(0,0,2),(0,1,1),(1,0,1),(1,1,0),
)
QUERY_EXPONENTS = MOMENT_EXPONENTS
QUERY_FACTORS = np.array([1.,1.,1.,1.,1.,1.,1.,2.,2.,2.],dtype=np.float64)
QUERY_ORDER = ("1","x","y","z","xx","yy","zz","2yz","2xz","2xy")


@dataclass(frozen=True)
class QuadricDerivativeResult:
    moments: np.ndarray
    jacobian: np.ndarray | None
    derivative_valid: bool
    event: str
    cut_x: float | None


def _box(bounds):
    b=np.asarray(bounds,dtype=np.float64)
    if b.shape!=(3,2) or not np.isfinite(b).all() or np.any(b[:,1]<=b[:,0]):
        raise ValueError("bounds must be finite [3,2] with positive extents")
    return b


def _q(q):
    value=np.asarray(q,dtype=np.float64)
    if value.shape!=(10,) or not np.isfinite(value).all() or np.linalg.norm(value)<=1e-14:
        raise ValueError("quadric coefficients must be a finite nonzero vector [10]")
    return value


def _powint(lo,hi,p):
    return (hi**(p+1)-lo**(p+1))/(p+1)


def _face_product_integral(a, yb, zb, exp_m, exp_q):
    ix=a**(exp_m[0]+exp_q[0])
    iy=_powint(yb[0],yb[1],exp_m[1]+exp_q[1])
    iz=_powint(zb[0],zb[1],exp_m[2]+exp_q[2])
    return ix*iy*iz


def plane_cut_box_moments_and_jacobian(q, bounds, density=1.0, *, event_tol=1e-10):
    """Exact M and shape Jacobian dM/dq for q=c+b_x*x>=0, b_x<0.

    All ten Jacobian columns are exact first-variation boundary integrals.
    A cut coincident with either x face is an event and returns jacobian=None.
    """
    q=_q(q); b=_box(bounds)
    rho=float(density)
    if not math.isfinite(rho) or rho<=0: raise ValueError("density must be positive")
    if q[2:] .any() or q[1]>=0:
        raise ValueError("analytic fixture requires q=(c,bx,0,...,0), bx<0")
    a=-q[0]/q[1]
    x0,x1=b[0]; yb=b[1]; zb=b[2]
    if a<=x0 or a>=x1:
        # Values are still exactly computable, but the derivative is not
        # admitted at a support/topology endpoint.
        cut=float(np.clip(a,x0,x1))
        M=np.zeros(10)
        for i,(px,py,pz) in enumerate(MOMENT_EXPONENTS):
            M[i]=rho*_powint(x0,cut,px)*_powint(*yb,py)*_powint(*zb,pz)
        return QuadricDerivativeResult(M,None,False,"box_face_coincidence_or_empty_full",a)
    M=np.empty(10)
    for i,(px,py,pz) in enumerate(MOMENT_EXPONENTS):
        M[i]=rho*_powint(x0,a,px)*_powint(*yb,py)*_powint(*zb,pz)
    grad_norm=abs(q[1])
    J=np.empty((10,10))
    for i,em in enumerate(MOMENT_EXPONENTS):
        for j,eq in enumerate(QUERY_EXPONENTS):
            J[i,j]=rho*QUERY_FACTORS[j]*_face_product_integral(a,yb,zb,em,eq)/grad_norm
    return QuadricDerivativeResult(M,J,True,"regular_transversal_plane_cut",a)


def classify_repeated_root_or_plane_event(q, bounds, *, grad_tol=1e-10, event_tol=1e-9):
    """Classify analytic plane and repeated-root fixtures.

    Quadratic-form rank deficiency alone is NOT an invalid event: a regular
    plane has rank-zero quadratic part but a nonzero boundary gradient.
    """
    q=_q(q); b=_box(bounds)
    # Polynomial convention uses doubled cross terms in QUERY_ORDER.
    A=np.array([[q[4],q[9],q[8]],[q[9],q[5],q[7]],[q[8],q[7],q[6]]])
    rank=int(np.linalg.matrix_rank(A,tol=grad_tol))
    if rank>0:
        # A stationary zero is a singular quadric candidate. Exact test is
        # intentionally conservative: identify a zero-valued stationary point.
        try:
            stationary=np.linalg.solve(2*A,-q[1:4])
            value=q[0]+q[1:4]@stationary+stationary@A@stationary
            gradient=q[1:4]+2*A@stationary
            inside=np.all(stationary>=b[:,0]-event_tol) and np.all(stationary<=b[:,1]+event_tol)
            if inside and np.linalg.norm(gradient)<=grad_tol and abs(float(value))<=event_tol:
                return {"valid":False,"event":"singular_repeated_root","quadratic_rank":rank}
        except np.linalg.LinAlgError:
            pass
    if np.linalg.norm(q[2:])<=grad_tol and q[1]<-grad_tol:
        a=-q[0]/q[1]
        if abs(a-b[0,0])<=event_tol or abs(a-b[0,1])<=event_tol:
            return {"valid":False,"event":"box_face_coincidence","quadratic_rank":rank}
        if b[0,0]<a<b[0,1]:
            return {"valid":True,"event":"regular_plane_even_if_quadratic_rank_zero","quadratic_rank":rank}
    return {"valid":False,"event":"unsupported_or_unclassified_quadric_event","quadratic_rank":rank}


def _intervals_positive_quadratic(A,B,C,lo,hi):
    """Intervals in [lo,hi] where A*x²+B*x+C >= 0."""
    roots=[]
    if abs(A)<1e-14:
        if abs(B)<1e-14: return [(lo,hi)] if C>=0 else []
        roots=[-C/B]
    else:
        disc=B*B-4*A*C
        if disc>=0:
            s=math.sqrt(max(0.0,disc))
            roots=[(-B-s)/(2*A),(-B+s)/(2*A)]
    cuts=[lo]+sorted(x for x in roots if lo<x<hi)+[hi]
    out=[]
    for l,r in zip(cuts[:-1],cuts[1:]):
        mid=(l+r)/2
        if A*mid*mid+B*mid+C>=0: out.append((l,r))
    return out


def numerical_quadric_box_moments(q,bounds,density=1.0,*,order=96):
    """High-order deterministic yz Gauss quadrature with exact x interval moments.

    Used only as an independent finite-difference evaluator for nearby quadrics.
    """
    q=_q(q); b=_box(bounds); rho=float(density)
    if not math.isfinite(rho) or rho<=0: raise ValueError("density must be positive")
    nodes,weights=np.polynomial.legendre.leggauss(order)
    ys=(b[1,1]+b[1,0])/2+(b[1,1]-b[1,0])*nodes/2
    zs=(b[2,1]+b[2,0])/2+(b[2,1]-b[2,0])*nodes/2
    wy=weights*(b[1,1]-b[1,0])/2
    wz=weights*(b[2,1]-b[2,0])/2
    out=np.zeros(10)
    for iy,y in enumerate(ys):
        for iz,z in enumerate(zs):
            A=q[4]
            B=q[1]+2*q[8]*z+2*q[9]*y
            C=q[0]+q[2]*y+q[3]*z+q[5]*y*y+q[6]*z*z+2*q[7]*y*z
            intervals=_intervals_positive_quadratic(A,B,C,b[0,0],b[0,1])
            w=rho*wy[iy]*wz[iz]
            for k,(px,py,pz) in enumerate(MOMENT_EXPONENTS):
                yz=(y**py)*(z**pz)
                out[k]+=w*yz*sum(_powint(l,r,px) for l,r in intervals)
    return out


def central_difference_jacobian(q,bounds,density=1.0,*,step=2e-5,order=96):
    q=_q(q)
    J=np.empty((10,10))
    for j in range(10):
        h=step*max(1.0,abs(q[j]))
        d=np.zeros(10);d[j]=h
        J[:,j]=(numerical_quadric_box_moments(q+d,bounds,density,order=order)
                -numerical_quadric_box_moments(q-d,bounds,density,order=order))/(2*h)
    return J


def tangent_project(q,gradient):
    q=_q(q); g=np.asarray(gradient,dtype=np.float64)
    if g.shape!=(10,) or not np.isfinite(g).all(): raise ValueError("gradient must be finite [10]")
    u=q/np.linalg.norm(q)
    return g-u*np.dot(u,g)


def sphere_retract(q,tangent_step):
    q=_q(q); d=np.asarray(tangent_step,dtype=np.float64)
    if d.shape!=(10,) or not np.isfinite(d).all(): raise ValueError("step must be finite [10]")
    u=q/np.linalg.norm(q)
    d=tangent_project(u,d)
    out=u+d
    n=np.linalg.norm(out)
    if n<=1e-14: raise ValueError("retraction reached zero")
    out=out/n
    if np.dot(out,u)<=0: raise ValueError("step crossed the projective sign boundary")
    return out


def sphere_newton_step(q,gradient,hessian,*,damping=1e-6):
    """Damped Riemannian Newton step in an orthonormal S^9 tangent basis."""
    q=_q(q); g=np.asarray(gradient,dtype=np.float64); H=np.asarray(hessian,dtype=np.float64)
    if g.shape!=(10,) or H.shape!=(10,10) or not np.isfinite(H).all() or not np.isfinite(g).all():
        raise ValueError("gradient/Hessian must have shapes [10] and [10,10]")
    if damping<0 or not math.isfinite(damping): raise ValueError("damping must be nonnegative")
    u=q/np.linalg.norm(q)
    # QR of a deterministic 10x9 seed, then remove the radial component.
    seed=np.eye(10)[:,1:]
    basis,_=np.linalg.qr(seed-u[:,None]*(u@seed)[None,:],mode="reduced")
    Ht=basis.T@((H+H.T)/2)@basis
    gt=basis.T@tangent_project(u,g)
    eigmin=float(np.linalg.eigvalsh(Ht).min())
    lam=max(float(damping),-eigmin+float(damping))
    step_t=-np.linalg.solve(Ht+lam*np.eye(9),gt)
    step=basis@step_t
    return sphere_retract(u,step),step,{"tangent_gradient_norm":float(np.linalg.norm(gt)),
                                        "damping_used":lam,"tangent_basis_orthogonality":float(np.linalg.norm(basis.T@u))}


def material_jacobian_from_query_jacobian(query_jacobian,design_map_jacobian,*,same_frame, same_retained_side,
                                           same_masks, same_units, derivative_valid):
    """Apply chain rule only when query and material parameter contracts match."""
    J=np.asarray(query_jacobian,dtype=np.float64)
    D=np.asarray(design_map_jacobian,dtype=np.float64)
    if J.ndim!=2 or J.shape[0]!=10 or D.ndim!=2 or D.shape[0]!=J.shape[1]:
        raise ValueError("incompatible query/design Jacobian shapes")
    if not derivative_valid: raise ValueError("query derivative is not admitted on this branch")
    if not all((same_frame,same_retained_side,same_masks,same_units)):
        raise ValueError("query Jacobian cannot be reinterpreted as material Jacobian: contract mismatch")
    return J@D
