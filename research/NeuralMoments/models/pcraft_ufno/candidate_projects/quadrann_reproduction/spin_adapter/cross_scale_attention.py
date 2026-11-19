"""Query-conditioned Cross-scale Attention over cached P-CRAFT scale summaries."""
from __future__ import annotations
import math
import torch
from torch import Tensor, nn


class QueryConditionedCrossScaleAttention(nn.Module):
    def __init__(self,width:int,*,heads:int=2,max_scales:int=16):
        super().__init__()
        if width<1 or heads<1 or width%heads: raise ValueError("width must be divisible by heads")
        if max_scales<1: raise ValueError("max_scales must be positive")
        self.width=width; self.heads=heads; self.dim=width//heads; self.max_scales=max_scales
        self.query_encoder=nn.Sequential(nn.Linear(10,width),nn.GELU(),nn.Linear(width,width))
        self.scale_id=nn.Embedding(max_scales,width)
        self.q=nn.Linear(width,width,bias=False); self.k=nn.Linear(width,width,bias=False)
        self.v=nn.Linear(width,width,bias=False)
        self.out=nn.Sequential(nn.LayerNorm(2*width),nn.Linear(2*width,width),nn.GELU(),nn.Linear(width,10))
        nn.init.zeros_(self.out[-1].weight); nn.init.zeros_(self.out[-1].bias)

    def forward(self,coefficients:Tensor,scale_features:Tensor,scale_ids:Tensor,
                scale_mask:Tensor|None=None,disabled_scale_ids:tuple[int,...]=()):
        if coefficients.ndim!=2 or coefficients.shape[-1]!=10: raise ValueError("coefficients must be [Q,10]")
        if scale_features.ndim!=2 or scale_features.shape[-1]!=self.width: raise ValueError("scale_features must be [S,width]")
        if scale_ids.ndim!=1 or len(scale_ids)!=len(scale_features): raise ValueError("scale_ids must be [S]")
        if len(scale_ids)==0: raise ValueError("at least one scale token is required")
        if not torch.isfinite(scale_features).all() or not torch.isfinite(coefficients).all():
            raise ValueError("cross-scale inputs must be finite")
        ids=scale_ids.to(device=scale_features.device,dtype=torch.long)
        if (ids<=0).any(): raise ValueError("scale IDs must be positive native resolutions")
        # Resolution buckets preserve scale identity while supporting arbitrary
        # configured grids such as 16/32/64/128 with a compact embedding table.
        id_buckets=torch.floor(torch.log2(ids.float())).long().clamp(0,self.max_scales-1)
        valid=torch.ones(len(ids),device=ids.device,dtype=torch.bool)
        if scale_mask is not None:
            mask=torch.as_tensor(scale_mask,device=ids.device,dtype=torch.bool)
            if mask.shape!=valid.shape: raise ValueError("scale_mask must have shape [S]")
            valid &= mask
        for sid in disabled_scale_ids: valid &= ids!=int(sid)
        if not valid.any(): raise ValueError("all cross scales are missing or disabled")
        qnorm=coefficients/torch.linalg.vector_norm(coefficients,dim=-1,keepdim=True).clamp_min(1e-30)
        qtoken=self.query_encoder(qnorm.to(device=scale_features.device,dtype=scale_features.dtype))
        stoken=scale_features+self.scale_id(id_buckets)
        q=self.q(qtoken).reshape(-1,self.heads,self.dim)
        k=self.k(stoken).reshape(-1,self.heads,self.dim)
        v=self.v(stoken).reshape(-1,self.heads,self.dim)
        scores=torch.einsum("qhd,shd->qhs",q,k)/math.sqrt(self.dim)
        scores=scores.masked_fill(~valid.view(1,1,-1),-torch.inf)
        weights=torch.softmax(scores.float(),dim=-1).to(scores.dtype)
        context=torch.einsum("qhs,shd->qhd",weights,v).reshape(-1,self.width)
        correction=self.out(torch.cat((qtoken,context),dim=-1))
        entropy=-(weights.float().clamp_min(1e-30)*weights.float().clamp_min(1e-30).log()).sum(-1).mean(-1)
        diagnostics={"cross_scale_weights":weights.mean(dim=1),"cross_scale_entropy":entropy,
            "cross_scale_effective_support":torch.exp(entropy),
            "cross_scale_valid_count":valid.sum().expand(len(coefficients))}
        return correction,diagnostics
