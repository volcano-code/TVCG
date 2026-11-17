"""PINO material v1: fixed physical measure and learned bounded transition width.

Experimental, untrained, single geometry. No M83 rasterization or legacy model
mutation. See the adjacent README for units, smoothing and memory limitations.
"""
from dataclasses import dataclass
import math
import numpy as np
import torch
from torch import Tensor, nn

from research.NeuralMoments.models.pcraft_ufno.spin_pino.pino_backbone import PINOFeatureEncoder

DEGREES = (0, 1, 1, 1, 2, 2, 2, 2, 2, 2)


@dataclass(frozen=True)
class MaterialGrid:
    density: Tensor             # [X,Y,Z], physical editable density, differentiable
    editable: Tensor            # disjoint binary voxel support
    fixed_density: Tensor       # immutable physical fixed density
    fixed: Tensor               # disjoint binary voxel support
    lower: tuple[float, float, float]
    upper: tuple[float, float, float]
    length: float               # physical length unit for dimensionless queries
    density_reference: float    # input scaling only; never replaces physical density

    def validate(self):
        rho = self.density
        if rho.ndim != 3 or any(n < 8 or n % 4 for n in rho.shape):
            raise ValueError('grid dimensions must be >=8 and divisible by4')
        if rho.dtype not in (torch.float32, torch.float64):
            raise ValueError('material integration requires float32 or float64')
        for value in (rho, self.editable, self.fixed_density, self.fixed):
            if value.shape != rho.shape or value.device != rho.device:
                raise ValueError('material fields must share shape and device')
            if not bool(torch.isfinite(value).all()) or bool((value < 0).any()):
                raise ValueError('material fields must be finite and nonnegative')
        for mask in (self.editable, self.fixed):
            if mask.requires_grad or not bool(((mask == 0) | (mask == 1)).all()):
                raise ValueError('support masks must be constant and binary')
        if bool((self.editable.to(rho.dtype) + self.fixed.to(rho.dtype) > 1).any()):
            raise ValueError('fixed and editable support must not overlap')
        if self.fixed_density.requires_grad:
            raise ValueError('fixed density cannot be optimized')
        if len(self.lower) != 3 or len(self.upper) != 3:
            raise ValueError('bounds need three world coordinates')
        if any(not math.isfinite(x) for x in (*self.lower, *self.upper)):
            raise ValueError('bounds must be finite')
        if any(b <= a for a, b in zip(self.lower, self.upper)):
            raise ValueError('upper bounds must exceed lower bounds')
        if any(not math.isfinite(x) or x <= 0 for x in (self.length, self.density_reference)):
            raise ValueError('unit references must be positive and finite')


@dataclass(frozen=True)
class EncodedMaterial:
    basis: Tensor               # dimensionless node basis [cell,node,10]
    moment_scale: Tensor        # L^degree
    mass: Tensor               # editable physical mass per node
    tau: Tensor                # positive per-cell, shared over queries/nodes
    fixed: Tensor              # physical fixed moments
    full: Tensor               # physical fixed + full editable moments


def basis10(x: Tensor) -> Tensor:
    a, b, c = x.unbind(-1)
    return torch.stack((torch.ones_like(a), a, b, c, a*a, b*b, c*c, b*c, a*c, a*b), -1)


class PINOMaterialV1(nn.Module):
    """Sole neural core is PINO; the positive integration measure is physical.

    Encoding retains a differentiable graph; re-encode after density/parameter
    updates. Do not reuse a backward-freed cache. No automatic voxel resampling.
    The legacy encoder is used via tensor pass-through with NEW channel semantics.
    """
    version = 'pino-material-v1'

    def __init__(self, *, width=8, modes=4, layers=2, order=2,
                 tau_min=.02, tau_max=.20):
        super().__init__()
        if not isinstance(order, int) or not 2 <= order <= 8:
            raise ValueError('Gauss order must be an integer from2 through8')
        if not 0 < tau_min < tau_max < 1:
            raise ValueError('require 0 < tau_min < tau_max <1 for exact endpoints')
        self.encoder = PINOFeatureEncoder(in_channels=4, width=width, modes=modes,
                                          n_layers=layers, grid_size=8)
        self.bandwidth = nn.Conv3d(width, 1, 1)
        self.order, self.tau_min, self.tau_max = order, tau_min, tau_max

    def encode(self, grid: MaterialGrid) -> EncodedMaterial:
        grid.validate()
        rho = grid.density
        if rho.device != self.bandwidth.weight.device:
            raise ValueError('model and material must share a device')
        editable = grid.editable.to(rho.dtype)
        fixed = grid.fixed.to(rho.dtype)
        features = torch.stack((rho*editable/grid.density_reference, editable,
                                grid.fixed_density.to(rho.dtype)*fixed/grid.density_reference, fixed))[None]
        _, field = self.encoder.forward_with_map(features, preserve_resolution=True)
        tau = self.tau_min + (self.tau_max-self.tau_min)*torch.sigmoid(self.bandwidth(field))
        tau = tau.reshape(-1, 1).to(rho.dtype)

        lower, upper = rho.new_tensor(grid.lower), rho.new_tensor(grid.upper)
        spacing = (upper-lower)/rho.new_tensor(rho.shape)
        axes = [lower[i]+(torch.arange(n, device=rho.device, dtype=rho.dtype)+.5)*spacing[i]
                for i, n in enumerate(rho.shape)]
        centers = torch.stack(torch.meshgrid(*axes, indexing='ij'), -1).reshape(-1, 3)
        abscissa, weights = np.polynomial.legendre.leggauss(self.order)
        a, w = rho.new_tensor(abscissa), rho.new_tensor(weights)
        offsets = torch.stack(torch.meshgrid(a,a,a,indexing='ij'),-1).reshape(-1,3)
        wx, wy, wz = torch.meshgrid(w,w,w,indexing='ij')
        volume = (wx*wy*wz).reshape(1,-1)*spacing.prod()/8
        unit_basis = basis10((centers[:,None,:]+offsets[None]*spacing/2)/grid.length)
        scales = rho.new_tensor(grid.length)**rho.new_tensor(DEGREES)
        mass = (rho*editable).reshape(-1,1)*volume
        fixed_mass = (grid.fixed_density.to(rho.dtype)*fixed).reshape(-1,1)*volume
        F = torch.einsum('cn,cnk->k',fixed_mass,unit_basis)*scales
        full = F + torch.einsum('cn,cnk->k',mass,unit_basis)*scales
        if not all(bool(torch.isfinite(t).all()) for t in (volume,unit_basis,scales,mass,F,full,tau)):
            raise ValueError('physical units overflow the integration dtype')
        return EncodedMaterial(unit_basis,scales,mass,tau,F,full)

    def evaluate(self, cache: EncodedMaterial, c: Tensor) -> Tensor:
        """Raw world-coordinate coefficients [query,10] -> physical ten moments.

        No convex potential is exposed. The bounded smooth indicator is an
        approximation to a hard cut; low Gauss order can underresolve its band.
        """
        if c.ndim != 2 or c.shape[1] != 10 or c.shape[0] == 0 or not c.is_floating_point():
            raise ValueError('raw query must be nonempty floating [query,10]')
        transformed = c.to(cache.basis)*cache.moment_scale
        if not bool(torch.isfinite(transformed).all()):
            raise ValueError('query or transformed query is nonfinite')
        maximum = transformed.abs().amax(-1,keepdim=True).detach()
        if bool((maximum == 0).any()):
            raise ValueError('zero query has no oriented boundary')
        scaled = transformed/maximum
        unit = scaled/torch.linalg.vector_norm(scaled,dim=-1,keepdim=True)
        B = unit.new_tensor([1,1,1,1,1,1,1,2,2,2])
        f = torch.einsum('cnk,qk->qcn',cache.basis,unit*B)
        t = ((f/cache.tau + 1)/2).clamp(0,1)
        alpha = t.pow(3)*(10+t*(-15+6*t))
        # Guard last-bit overshoot of polynomial evaluation, retaining [0,1].
        alpha = alpha.clamp(0,1)
        return cache.fixed + torch.einsum('qcn,cn,cnk->qk',alpha,cache.mass,cache.basis)*cache.moment_scale

    def forward(self, grid: MaterialGrid, c: Tensor) -> Tensor:
        return self.evaluate(self.encode(grid),c)

    def jacobian_input(self, cache: EncodedMaterial, c: Tensor, *, create_graph=False) -> Tensor:
        """Return [query,moment,raw coefficient]; differentiable if requested.

        Queries are independent, so summing a moment across queries extracts
        their individual derivatives. Autograd mode is enabled for this API.
        """
        with torch.enable_grad():
            query = c if c.requires_grad else c.detach().clone().requires_grad_()
            moments = self.evaluate(cache,query)
            return torch.stack([torch.autograd.grad(moments[:,i].sum(),query,
                               retain_graph=True,create_graph=create_graph)[0]
                               for i in range(10)],dim=1)
