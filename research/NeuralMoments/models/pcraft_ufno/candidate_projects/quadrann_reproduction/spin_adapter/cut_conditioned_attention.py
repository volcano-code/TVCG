"""Experimental cut-conditioned local cross-attention for Compact PINO V2.4.

Opt-in research primitive; not wired into the V2.3 production path. It conditions
only compacted active material tokens on descriptors of the current finite cut.
"""
from __future__ import annotations
from dataclasses import dataclass
import torch
from torch import Tensor, nn

@dataclass(frozen=True)
class CutAttentionStats:
    active_queries: int
    cut_tokens: int
    attention_elements: int

class CutConditionedCrossAttention(nn.Module):
    """Cross-attend active material tokens to a small set of cut-condition tokens.

    Inputs:
      material_tokens: [B,N,D]
      cut_tokens: [B,S,D]
      active_mask: [B,N] bool
      cut_mask: [B,S] bool; True denotes a valid cut token.

    Only active material tokens are gathered into attention. Inactive tokens
    never enter Q/K/V attention work and are exact pass-through values. Complexity
    is proportional to sum_b(N_active_b*S_b*D), not N_all^2*D.

    The zero residual gate gives an exact identity conditioned-feature path at
    initialization. The zero-initialized response projection gives zero response
    delta initially while receiving a useful gradient for nonzero targets.
    Keep cut_tokens attached to autograd: their derivative belongs in dM/dc.
    """
    def __init__(self, width: int = 192, heads: int = 4,
                 output_dim: int = 2, dropout: float = 0.0):
        super().__init__()
        if width < 1 or heads < 1 or width % heads:
            raise ValueError("width must be positive and divisible by heads")
        if output_dim < 1 or not 0.0 <= dropout < 1.0:
            raise ValueError("invalid output_dim or dropout")
        self.width, self.heads, self.output_dim = width, heads, output_dim
        self.norm_q, self.norm_kv = nn.LayerNorm(width), nn.LayerNorm(width)
        self.attention = nn.MultiheadAttention(width, heads, dropout=dropout, batch_first=True)
        self.ffn = nn.Sequential(nn.LayerNorm(width), nn.Linear(width, 2*width),
                                 nn.GELU(), nn.Linear(2*width, width))
        self.response_delta = nn.Sequential(nn.LayerNorm(width), nn.Linear(width, output_dim))
        self.residual_gate = nn.Parameter(torch.zeros(()))
        nn.init.zeros_(self.response_delta[-1].weight)
        nn.init.zeros_(self.response_delta[-1].bias)

    def forward(self, material_tokens: Tensor, cut_tokens: Tensor,
                active_mask: Tensor, cut_mask: Tensor | None = None,
                *, need_weights: bool = False):
        if material_tokens.ndim != 3 or cut_tokens.ndim != 3:
            raise ValueError("material_tokens and cut_tokens must be rank-3")
        b, n, d = material_tokens.shape
        bc, s, dc = cut_tokens.shape
        if b != bc or d != self.width or dc != self.width or s < 1:
            raise ValueError("incompatible material/cut token shapes")
        if active_mask.shape != (b,n) or active_mask.dtype != torch.bool:
            raise ValueError("active_mask must be Boolean [B,N]")
        if cut_mask is None:
            cut_mask = torch.ones((b,s), dtype=torch.bool, device=cut_tokens.device)
        if cut_mask.shape != (b,s) or cut_mask.dtype != torch.bool:
            raise ValueError("cut_mask must be Boolean [B,S]")
        if not bool(cut_mask.any(dim=1).all()):
            raise ValueError("each sample needs at least one valid cut token")
        if material_tokens.device != cut_tokens.device or material_tokens.dtype != cut_tokens.dtype:
            raise ValueError("material and cut tokens must share device and dtype")

        conditioned = material_tokens.clone()
        delta = material_tokens.new_zeros((b,n,self.output_dim))
        attention_weights = [] if need_weights else None
        active_total = 0
        cut_total = 0
        attention_elements = 0
        gate = torch.tanh(self.residual_gate)

        # Per-sample compaction ensures inactive material tokens do not enter MHA.
        for batch_index in range(b):
            q_index = torch.nonzero(active_mask[batch_index], as_tuple=False).flatten()
            kv_index = torch.nonzero(cut_mask[batch_index], as_tuple=False).flatten()
            n_active, n_cut = int(q_index.numel()), int(kv_index.numel())
            active_total += n_active
            cut_total += n_cut
            attention_elements += n_active * n_cut
            if n_active == 0:
                if need_weights:
                    attention_weights.append(None)
                continue
            q = self.norm_q(material_tokens[batch_index:batch_index+1].index_select(1,q_index))
            kv = self.norm_kv(cut_tokens[batch_index:batch_index+1].index_select(1,kv_index))
            attended, weights = self.attention(q,kv,kv,need_weights=need_weights,
                                               average_attn_weights=False)
            updated = q.squeeze(0) + gate * self.ffn(attended).squeeze(0)
            response = self.response_delta(updated)
            conditioned[batch_index] = conditioned[batch_index].index_copy(0,q_index,updated)
            delta[batch_index] = delta[batch_index].index_copy(0,q_index,response)
            if need_weights:
                attention_weights.append(weights)

        stats = CutAttentionStats(active_total, cut_total, attention_elements)
        if need_weights:
            return conditioned, delta, stats, attention_weights
        return conditioned, delta, stats
