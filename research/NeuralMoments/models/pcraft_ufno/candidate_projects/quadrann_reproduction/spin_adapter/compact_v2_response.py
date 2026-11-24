"""Mass-conservative token response for Compact PINO V2.

Every native RLE interval is assigned or split exactly at token boundaries.
One scalar soft retained fraction multiplies each token's exact RLE moments.
This is a learned approximation of partial cuts, not a native mesh cutter.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
import numpy as np
import torch
from torch import nn, Tensor


@dataclass(frozen=True)
class PhysicalTokens:
    inverse: Tensor
    source_index: Tensor
    segment_fraction: Tensor
    native_intervals: int
    moments: Tensor
    lower: Tensor
    upper: Tensor
    coordinate_scales: Tensor


def physical_tokens(geometry, grid: int, *, device) -> PhysicalTokens:
    """Split native RLE runs at token boundaries and conserve their moments."""
    if isinstance(grid, bool) or not isinstance(grid, int) or grid < 1:
        raise ValueError("token grid must be a positive integer")
    xyz = np.asarray(geometry["xyz"], dtype=np.float64)
    extent = np.asarray(geometry["cell_extent"], dtype=np.float64)
    mass = np.asarray(geometry["mass"], dtype=np.float64)
    basis = np.asarray(geometry["moment_basis"], dtype=np.float64)
    if (xyz.ndim != 2 or xyz.shape[1] != 3 or len(xyz) == 0
            or extent.shape != xyz.shape or mass.shape != (len(xyz),)
            or basis.shape != (len(xyz), 10)):
        raise ValueError("invalid native interval arrays")
    if not all(np.isfinite(x).all() for x in (xyz, extent, mass, basis)) or (mass <= 0).any() or (extent <= 0).any():
        raise ValueError("invalid physical interval data")
    origin = np.asarray(geometry["origin"], dtype=np.float64)
    span = int(np.asarray(geometry["native_resolution"])) * float(geometry["h"])
    if origin.shape != (3,) or not np.isfinite(origin).all() or not math.isfinite(span) or span <= 0:
        raise ValueError("invalid native coordinate frame")
    native_count=len(xyz)
    side=int(np.asarray(geometry["native_resolution"]))
    columns=geometry.get("columns")
    if columns is not None:
        columns=np.asarray(columns)
        if (columns.shape!=(native_count,4) or columns.dtype.kind not in "iu"
                or side%grid or (columns[:,:3]<0).any() or (columns[:,:2]>=side).any()
                or (columns[:,3]>side).any() or (columns[:,3]<=columns[:,2]).any()):
            raise ValueError("RLE token grid must divide side and columns must be valid [N,4]")
        columns=columns.astype(np.int64,copy=False)
        stride=side//grid
        first=columns[:,2]//stride;last=(columns[:,3]-1)//stride
        counts=last-first+1
        parent=np.repeat(np.arange(native_count),counts)
        zbin=np.repeat(first,counts)+np.arange(len(parent))-np.repeat(np.cumsum(counts)-counts,counts)
        seg_lo=np.maximum(columns[parent,2],zbin*stride)
        seg_hi=np.minimum(columns[parent,3],(zbin+1)*stride)
        fraction=(seg_hi-seg_lo)/(columns[parent,3]-columns[parent,2])
        h=float(geometry["h"])
        z0=origin[2]+seg_lo*h;z1=origin[2]+seg_hi*h;z=(z0+z1)*.5
        xyz=xyz[parent].copy();extent=extent[parent].copy();basis=basis[parent].copy()
        xyz[:,2]=z;extent[:,2]=z1-z0;mass=mass[parent]*fraction
        basis[:,3]=z;basis[:,6]=(z0*z0+z0*z1+z1*z1)/3
        basis[:,7]=xyz[:,1]*z;basis[:,8]=xyz[:,0]*z
        ids=np.column_stack((columns[parent,:2]//stride,zbin))
    else:
        # Generic rectangular-prism fixtures have no integer RLE columns.
        # Production native RLE always follows the exact split path above.
        parent=np.arange(native_count);fraction=np.ones(native_count)
        ids=np.clip(np.floor((xyz-origin)/span*grid).astype(np.int64),0,grid-1)
    linear = (ids[:,0]*grid+ids[:,1])*grid+ids[:,2]
    active, inverse = np.unique(linear, return_inverse=True)
    n = len(active)
    moments = np.zeros((n,10),dtype=np.float64)
    np.add.at(moments,inverse,mass[:,None]*basis)
    lo = np.full((n,3),np.inf); hi = np.full((n,3),-np.inf)
    np.minimum.at(lo,inverse,xyz-extent/2)
    np.maximum.at(hi,inverse,xyz+extent/2)
    length = max(float((hi.max(0)-lo.min(0)).max()),np.finfo(np.float64).eps)
    scales = length ** np.array([0,1,1,1,2,2,2,2,2,2],dtype=np.float64)
    lo,hi=lo/length,hi/length
    low=[np.ones(n)]; high=[np.ones(n)]
    for i in range(3):
        low.append(lo[:,i]); high.append(hi[:,i])
    for i in range(3):
        a,b=lo[:,i],hi[:,i]
        low.append(np.where((a<=0)&(b>=0),0.,np.minimum(a*a,b*b)))
        high.append(np.maximum(a*a,b*b))
    for i,j in ((1,2),(0,2),(0,1)):
        corners=np.stack([lo[:,i]*lo[:,j],lo[:,i]*hi[:,j],hi[:,i]*lo[:,j],hi[:,i]*hi[:,j]])
        low.append(corners.min(0));high.append(corners.max(0))
    def tensor(x):return torch.as_tensor(x,device=device)
    return PhysicalTokens(tensor(inverse),tensor(parent),tensor(fraction),native_count,tensor(moments),tensor(np.stack(low,1)),tensor(np.stack(high,1)),tensor(scales))


def pool_features(features: Tensor, interval_mass: Tensor, tokens: PhysicalTokens) -> Tensor:
    """Mass-weighted learned features; gradient returns to every interval."""
    if features.ndim != 2 or len(features) != tokens.native_intervals or interval_mass.shape != (len(features),):
        raise ValueError("invalid interval feature/mass shapes")
    pool=features.new_zeros(len(tokens.moments),features.shape[1])
    den=features.new_zeros(len(tokens.moments))
    # Avoid materializing a [split_segments, feature_width] table.
    weights=interval_mass.to(features)
    for start in range(0,len(tokens.inverse),8192):
        stop=min(start+8192,len(tokens.inverse))
        src=tokens.source_index[start:stop]; dst=tokens.inverse[start:stop]
        weight=weights[src]*tokens.segment_fraction[start:stop].to(features)
        pool.index_add_(0,dst,features[src]*weight[:,None])
        den.index_add_(0,dst,weight)
    return pool/den[:,None].clamp_min(torch.finfo(features.dtype).tiny)


@dataclass(frozen=True)
class ResponseMemory:
    moments: Tensor
    lower: Tensor
    upper: Tensor
    mean_basis: Tensor
    coordinate_scales: Tensor
    local: Tensor
    keys: Tensor
    values: Tensor
    log_prior: Tensor


class CutAwareMomentResponse(nn.Module):
    """Even-query cross attention -> odd cut correction -> one bounded integral.

    All trainable interactions occur before the physical moment readout. No
    separate M or J head, no additive free moment residual, no learned mass.
    """
    def __init__(self, feature_width: int, width: int=64, heads: int=4):
        super().__init__()
        if min(feature_width,width,heads)<1 or width%heads:
            raise ValueError("invalid response dimensions")
        self.width=width;self.heads=heads;self.dim=width//heads
        self.token_encoder=nn.Sequential(nn.LayerNorm(feature_width+30),nn.Linear(feature_width+30,width),nn.GELU())
        self.query_encoder=nn.Sequential(nn.Linear(100,width),nn.GELU(),nn.Linear(width,width))
        self.key=nn.Linear(width,width,bias=False)
        self.value=nn.Linear(width,width,bias=False)
        self.local=nn.Linear(width,width)
        self.query=nn.Linear(width,width)
        self.context=nn.Linear(width,width)
        self.cut=nn.Sequential(nn.GELU(),nn.Linear(width,10))
        nn.init.normal_(self.cut[-1].weight,std=.02)
        nn.init.zeros_(self.cut[-1].bias)
        self.register_buffer("native_B",torch.tensor([1.,1.,1.,1.,1.,1.,1.,2.,2.,2.],dtype=torch.float64))

    def prepare(self, pooled: Tensor, tokens: PhysicalTokens) -> ResponseMemory:
        mean=tokens.moments/tokens.moments[:,:1]/tokens.coordinate_scales
        f=torch.cat([pooled,mean.to(pooled),tokens.lower.to(pooled),tokens.upper.to(pooled)],-1)
        z=self.token_encoder(f)
        log_prior=(tokens.moments[:,0]/tokens.moments[:,0].sum()).log().to(z)
        return ResponseMemory(tokens.moments,tokens.lower,tokens.upper,mean,tokens.coordinate_scales,
                              self.local(z),self.key(z).reshape(-1,self.heads,self.dim),
                              self.value(z).reshape(-1,self.heads,self.dim),log_prior)

    def fractions(self, memory: ResponseMemory, coefficients: Tensor) -> Tensor:
        if coefficients.ndim!=2 or coefficients.shape[1]!=10 or not len(coefficients):
            raise ValueError("coefficients must be nonempty [Q,10]")
        c=coefficients.to(memory.moments)
        magnitude=c.abs().amax(-1,keepdim=True)
        if not torch.isfinite(c).all() or (magnitude<=0).any():
            raise ValueError("coefficients must be finite and nonzero")
        t=(c/magnitude)*self.native_B.to(c)*memory.coordinate_scales
        t=t/torch.linalg.vector_norm(t,dim=-1,keepdim=True)
        # t t^T is invariant to c -> -c. The learned correction is odd by
        # contraction with t, so complementary cuts share one response.
        even=(t[:,:,None]*t[:,None,:]).reshape(-1,100)
        q=self.query_encoder(even.to(memory.local))
        qh=q.reshape(-1,self.heads,self.dim)
        scores=torch.einsum("qhd,khd->qhk",qh,memory.keys)/math.sqrt(self.dim)+memory.log_prior
        context=torch.einsum("qhk,khd->qhd",scores.softmax(-1),memory.values).reshape(-1,self.width)
        d=self.cut(memory.local[None]+self.query(q)[:,None]+self.context(context)[:,None]).to(t)
        # Interval arithmetic in normalized monomial coordinates. Each token
        # bound covers the entire support of its native interval segments.
        midpoint=(memory.lower+memory.upper)*.5
        radius=(memory.upper-memory.lower)*.5
        center=t@midpoint.T
        # A smooth upper bound on |t| avoids artificial derivative kinks
        # when plane/restricted-chart coefficients cross zero. The enclosure
        # is conservative (slightly wider than exact interval arithmetic).
        bound=(t.square()+1e-12).sqrt()@radius.T
        low=center-bound;high=center+bound
        span=high-low
        interior=(low<0)&(high>0)
        # Do not even divide by a tiny width for saturated cells: masked
        # huge intermediate derivatives can otherwise contaminate AD.
        safe=torch.where(interior,span,torch.ones_like(span))
        u=torch.where(interior,high/safe,torch.full_like(high,.5)).clamp(0.,1.)
        mean=t@memory.mean_basis.T
        bias=torch.where(interior,(mean-(high+low)*.5)/safe,torch.zeros_like(span)).clamp(-.5,.5)
        delta=2*torch.tanh((d*t[:,None]).sum(-1)+2*bias)
        a=u.square()*delta.exp();b=(1-u).square()*(-delta).exp()
        p=a/(a+b)
        endpoint=torch.where((low>=0)&(high>0),torch.ones_like(p),
                            torch.where((high<=0)&(low<0),torch.zeros_like(p),torch.full_like(p,.5)))
        return torch.where(interior,p,endpoint)

    def forward(self, memory: ResponseMemory, fixed_moments: Tensor, coefficients: Tensor) -> Tensor:
        return fixed_moments[None]+self.fractions(memory,coefficients)@memory.moments
