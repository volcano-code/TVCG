"""Mass-aware, chunked Geometry–Physics Cross Attention (Phase 2.8.5.1)."""
from __future__ import annotations
import math
import torch
from torch import Tensor, nn
from torch.utils.checkpoint import checkpoint


class GeometryPhysicsCrossAttention(nn.Module):
    """Quadric queries attend to cached physical interval tokens.

    Interval processing uses online log-sum-exp, giving bounded O(Q*H*C)
    score memory for interval chunk C. The normalized physical mass is a
    log-prior in attention, not an unnormalized arbitrary token weight.
    """
    def __init__(self,width:int,*,heads:int=2,chunk_intervals:int=1024,
                 hidden:int|None=None,output_scale:float=.05):
        super().__init__()
        if width<1 or heads<1 or width%heads:
            raise ValueError("width must be positive and divisible by heads")
        if chunk_intervals<1 or output_scale<0:
            raise ValueError("invalid attention chunk or output scale")
        self.width=int(width); self.heads=int(heads); self.dim=width//heads
        self.chunk_intervals=int(chunk_intervals); self.output_scale=float(output_scale)
        hidden=int(hidden or width)
        self.query_encoder=nn.Sequential(nn.Linear(10,hidden),nn.GELU(),nn.Linear(hidden,width))
        self.query=nn.Linear(width,width,bias=False)
        self.key=nn.Linear(width,width,bias=False)
        self.value=nn.Linear(width,width,bias=False)
        self.output=nn.Sequential(nn.LayerNorm(2*width),nn.Linear(2*width,hidden),
                                  nn.GELU(),nn.Linear(hidden,10))
        nn.init.zeros_(self.output[-1].weight)
        nn.init.zeros_(self.output[-1].bias)

    def forward(self,coefficients:Tensor,interval_features:Tensor,
                interval_mass:Tensor)->tuple[Tensor,dict[str,Tensor]]:
        if coefficients.ndim!=2 or coefficients.shape[1]!=10:
            raise ValueError("coefficients must have shape [Q,10]")
        if interval_features.ndim!=2 or interval_features.shape[1]!=self.width:
            raise ValueError("interval_features must have shape [N,width]")
        if interval_mass.ndim!=1 or interval_mass.shape[0]!=len(interval_features):
            raise ValueError("interval_mass must have shape [N]")
        if len(interval_features)==0: raise ValueError("geometry has no intervals")
        if not torch.isfinite(coefficients).all():
            raise ValueError("attention coefficients must be finite")
        # Avoid torch.isfinite(interval_features) over the full 1M+ interval
        # matrix: its temporary boolean tensor can itself require hundreds of
        # MiB. Validate each already-bounded chunk immediately before use.
        if not torch.isfinite(interval_mass).all() or (interval_mass<=0).any():
            raise ValueError("physical interval masses must be finite and positive")
        dtype=interval_features.dtype; device=interval_features.device
        qnorm=coefficients/torch.linalg.vector_norm(coefficients,dim=-1,keepdim=True).clamp_min(1e-30)
        qtoken=self.query_encoder(qnorm.to(device=device,dtype=dtype))
        q=self.query(qtoken).reshape(-1,self.heads,self.dim)
        log_mass=torch.log(interval_mass.to(dtype=dtype).clamp_min(1e-30)) - torch.log(
            interval_mass.to(dtype=dtype).sum().clamp_min(1e-30)
        )
        contexts=[]; entropies=[]
        q_chunk=1
        scale=1.0/math.sqrt(self.dim)
        for q0 in range(0,len(coefficients),q_chunk):
            qc=q[q0:q0+q_chunk]
            running_max=torch.full((len(qc),self.heads,1),-torch.inf,device=device,dtype=dtype)
            denom=torch.zeros_like(running_max)
            value_sum=torch.zeros((len(qc),self.heads,self.dim),device=device,dtype=dtype)
            score_sum=torch.zeros_like(running_max)
            for n0 in range(0,len(interval_features),self.chunk_intervals):
                n1=min(n0+self.chunk_intervals,len(interval_features))
                feature_chunk=interval_features[n0:n1]
                if not torch.isfinite(feature_chunk).all():
                    raise ValueError(
                        f"attention interval features must be finite (chunk {n0}:{n1})"
                    )
                log_mass_chunk=log_mass[n0:n1]

                def project_score_chunk(q_chunk_input, feature_input, log_mass_input):
                    key_chunk=self.key(feature_input).reshape(-1,self.heads,self.dim)
                    value_chunk=self.value(feature_input).reshape(-1,self.heads,self.dim)
                    scores=torch.einsum("qhd,nhd->qhn",q_chunk_input,key_chunk)*scale
                    scores=scores+log_mass_input.view(1,1,-1)
                    local_max=scores.amax(dim=-1,keepdim=True)
                    weights=torch.exp(scores-local_max)
                    local_denom=weights.sum(dim=-1,keepdim=True)
                    local_value=torch.einsum("qhn,nhd->qhd",weights,value_chunk)
                    local_score=(weights*scores).sum(dim=-1,keepdim=True)
                    return local_max,local_denom,local_value,local_score

                # The caller checkpoints the complete attention-plus-query-
                # Jacobian function. Avoid nested per-interval checkpoints here:
                # nested checkpoint recomputation under create_graph=True caused
                # repeated replay of thousands of chunks and exhausted memory.
                chunk_max,chunk_denom,chunk_value,chunk_score=project_score_chunk(
                    qc,feature_chunk,log_mass_chunk
                )
                next_max=torch.maximum(running_max,chunk_max)
                old_scale=torch.exp(running_max-next_max)
                chunk_scale=torch.exp(chunk_max-next_max)
                denom=denom*old_scale+chunk_denom*chunk_scale
                value_sum=value_sum*old_scale+chunk_value*chunk_scale
                score_sum=score_sum*old_scale+chunk_score*chunk_scale
                running_max=next_max
            context=value_sum/denom.clamp_min(1e-30)
            log_z=running_max+torch.log(denom.clamp_min(1e-30))
            entropy=(log_z-score_sum/denom.clamp_min(1e-30)).squeeze(-1).clamp_min(0.)
            contexts.append(context.reshape(len(qc),self.width))
            entropies.append(entropy)
        context=torch.cat(contexts,dim=0)
        entropy=torch.cat(entropies,dim=0)
        correction=self.output(torch.cat((qtoken,context),dim=-1))*self.output_scale
        diagnostics={
            "attention_entropy":entropy.mean(dim=-1),
            "attention_effective_support":torch.exp(entropy).mean(dim=-1),
            "attention_mass_sum":interval_mass.sum().expand(len(coefficients)),
            "attention_query_norm":torch.linalg.vector_norm(coefficients,dim=-1),
        }
        return correction,diagnostics
