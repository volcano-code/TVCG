"""DensiCrafter-inspired bounded density design, adapted to fixed Spin-It support.

Independent implementation. Upstream provenance: idvxlab/DensiCrafter,
21e6d8dc646129a9b21a38c2d5bee5bef2704d98 (MIT), obj.py Rho.opt.
Unlike its basal-domain growth/full-grid filter, this adapter only removes
material and normalizes smoothing over editable neighbors. No teacher/network.
"""
from dataclasses import replace
import torch
import torch.nn.functional as F
from research.NeuralMoments.experiments.pino_material_v1.model import MaterialGrid


def bounded_material(base: MaterialGrid, logits: torch.Tensor, *, smoothing=True) -> MaterialGrid:
    """Map logits to rho_base*beta, preserving immutable fixed material.

    rho_base is a frozen per-cell physical density upper bound. beta is a
    dimensionless material fraction, not a second occupancy mask. The returned
    density already includes beta; PINO's binary support is applied once.
    """
    base.validate()
    if base.density.requires_grad:
        raise ValueError('base density must be constant; optimize logits instead')
    if (logits.shape != base.density.shape or logits.device != base.density.device
            or logits.dtype != base.density.dtype or not torch.isfinite(logits).all()):
        raise ValueError('finite logits must share base density shape, device and dtype')
    mask=base.editable.to(logits.dtype)
    fraction=torch.sigmoid(logits)*mask
    if smoothing:
        kernel=logits.new_ones((1,1,3,3,3))
        count=F.conv3d(mask[None,None],kernel,padding=1)
        fraction=(F.conv3d(fraction[None,None],kernel,padding=1)/count.clamp_min(1))[0,0]
    return replace(base,density=base.density*fraction*mask)
