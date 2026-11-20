"""Index native neighbors through their source tiles without dense full-table adjoints."""
from dataclasses import dataclass

import numpy as np
import torch


@dataclass(frozen=True)
class NeighborTile:
    sources: tuple[int, ...]
    indices: torch.Tensor

    def gather_source_table(self, values):
        if len(self.sources) == 1:
            return values[self.sources[0]]
        return torch.cat([values[i] for i in self.sources], dim=0)


def neighbor_tiles(neighbors, chunk, device):
    # Neighbor order within each target is unchanged, including self padding.
    # The (possibly short) final source tile is always last in sorted order.
    for start in range(0, len(neighbors), chunk):
        rows = neighbors[start:start+chunk]
        source_ids, inverse = np.unique(rows // chunk, return_inverse=True)
        local = inverse.reshape(rows.shape) * chunk + rows % chunk
        yield NeighborTile(tuple(map(int, source_ids)), torch.as_tensor(
            local.astype(np.int32), device=device))


class _SampleMapAdjoint(torch.autograd.Function):
    """Adjoint of trilinear sampling at fixed geometry coordinates."""
    @staticmethod
    def forward(ctx, gradient, grid, shape):
        ctx.save_for_backward(grid)
        # grid_sampler's map gradient depends only on grid, never on map values.
        dummy = gradient.new_zeros(1).expand(shape)
        return torch.ops.aten.grid_sampler_3d_backward(
            gradient, dummy, grid, 0, 1, True, [True, False])[0]

    @staticmethod
    def backward(ctx, gradient):
        grid, = ctx.saved_tensors
        return _SampleMap.apply(gradient, grid), None, None


class _SampleMap(torch.autograd.Function):
    @staticmethod
    def forward(ctx, feature_map, grid):
        ctx.save_for_backward(grid)
        ctx.map_shape = feature_map.shape
        return torch.nn.functional.grid_sample(feature_map,grid,mode="bilinear",
                                               padding_mode="border",align_corners=True)

    @staticmethod
    def backward(ctx, gradient):
        grid, = ctx.saved_tensors
        return _SampleMapAdjoint.apply(gradient,grid,ctx.map_shape),None


def sample_native_intervals(feature_map, coordinates):
    """One linear sampling operation and one map adjoint per resolution."""
    grid=coordinates.to(feature_map.dtype).reshape(1,-1,1,1,3)
    if grid.requires_grad:
        raise ValueError("native RLE sampling coordinates must be fixed")
    return _SampleMap.apply(feature_map,grid).reshape(feature_map.shape[1],-1).T
