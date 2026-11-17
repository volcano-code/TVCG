"""Multi-scale P-CRAFT feature adapter for SQNO.

Reuses the existing PcraftNativeMomentOperator spatial encoder and
interval feature implementation. Its original query_head and predict
methods are deliberately never invoked.

The adapter returns:
- per-interval local features;
- global multi-scale context;
- global scale weights;
- interval descriptors and normalized coordinates.
"""
from __future__ import annotations

import torch
from torch import nn
import numpy as np

from research.NeuralMoments.models.pcraft_ufno.spin_pino.native_pcraft import (
    PcraftNativeMomentOperator,
    make_config,
    geometry_from_columns,
    conservative_coarse_grid,
)


class PcraftFeatureAdapter(nn.Module):
    def __init__(
        self,
        preset="xl_ms",
        min_native_resolution=512,
        chunk_intervals=512,
    ):
        super().__init__()

        if chunk_intervals < 1:
            raise ValueError("chunk_intervals must be positive")

        config = make_config(
            preset,
            min_native_resolution=min_native_resolution,
        )

        if not config.multiscale_enabled:
            raise ValueError("P-CRAFT multiscale must be enabled")

        if tuple(config.multiscale_grids) != (32, 64, 128):
            raise ValueError("SQNO v1.1 requires 32/64/128 scales")

        self.operator = PcraftNativeMomentOperator(config)
        self.chunk_intervals = int(chunk_intervals)

        # This adapter uses only the geometry encoder, never the
        # original task-specific query head.
        if hasattr(self.operator, "query_head"):
            self.operator.query_head.requires_grad_(False)

    def forward(self, geometry):
        if "xyz" not in geometry:
            geometry = geometry_from_columns(geometry)

        op = self.operator
        config = op.config

        xyz_np = np.asarray(geometry["xyz"], dtype=np.float64)
        mass_np = np.asarray(geometry["mass"], dtype=np.float64)
        extent_np = np.asarray(geometry["cell_extent"], dtype=np.float64)
        basis_np = np.asarray(geometry["moment_basis"], dtype=np.float64)

        n = len(xyz_np)

        if n == 0:
            raise ValueError("geometry has no RLE intervals")

        if xyz_np.shape != (n, 3):
            raise ValueError("invalid interval coordinates")
        if mass_np.shape != (n,) or (mass_np <= 0).any():
            raise ValueError("invalid interval masses")
        if extent_np.shape != (n, 3):
            raise ValueError("invalid interval extents")
        if basis_np.shape != (n, 10):
            raise ValueError("invalid ten-dimensional moment basis")

        side = int(np.asarray(geometry["native_resolution"]))
        if side < config.min_native_resolution:
            raise ValueError("native resolution below P-CRAFT contract")

        parameter = next(op.parameters())
        device = parameter.device
        dtype = parameter.dtype

        xyz = torch.as_tensor(
            xyz_np.copy(), device=device, dtype=torch.float64
        )
        mass = torch.as_tensor(
            mass_np.copy(), device=device, dtype=torch.float64
        )
        extent = torch.as_tensor(
            extent_np.copy(), device=device, dtype=torch.float64
        )
        basis = torch.as_tensor(
            basis_np.copy(), device=device, dtype=torch.float64
        )

        fixed = torch.as_tensor(
            np.asarray(geometry["fixed_moments"], dtype=np.float64).copy(),
            device=device,
        )
        original = torch.as_tensor(
            np.asarray(geometry["original_moments"], dtype=np.float64).copy(),
            device=device,
        )

        design = mass @ basis

        lo = (xyz - extent / 2).amin(0)
        hi = (xyz + extent / 2).amax(0)

        length = (hi - lo).max().clamp_min(
            torch.finfo(torch.float64).eps
        )

        powers = torch.tensor(
            [0., 1., 1., 1., 2., 2., 2., 2., 2., 2.],
            device=device,
            dtype=torch.float64,
        )

        coord_scale = length ** powers
        moment_scale = (fixed[0] + mass.sum()) * coord_scale

        if not torch.allclose(
            fixed + design,
            original,
            rtol=1e-8,
            atol=float(moment_scale.max()) * 1e-10,
        ):
            raise ValueError("fixed + editable moment contract failed")

        # Exact same 36D statistics as native P-CRAFT encode().
        statistics = torch.cat((
            original / moment_scale,
            fixed / moment_scale,
            design / moment_scale,
            (lo + hi) / (2 * length),
            (hi - lo) / length,
        )).to(dtype)[None]

        # Exact same 13D RLE interval descriptor.
        descriptor = torch.cat((
            basis / coord_scale[None],
            extent / length,
        ), dim=-1).to(dtype)

        mass_weights = (mass / mass.sum()).to(dtype)

        # Generate conservative 32/64/128 P-CRAFT context grids.
        scale_maps = []
        scale_globals = []

        for grid_size in config.multiscale_grids:
            if (
                grid_size == config.grid_size
                and geometry.get("pcraft_grid") is not None
            ):
                grid_np = np.asarray(
                    geometry["pcraft_grid"],
                    dtype=np.float32,
                )
            else:
                grid_np = conservative_coarse_grid(
                    geometry, grid_size
                )

            expected = (4, grid_size, grid_size, grid_size)
            if grid_np.shape != expected:
                raise ValueError(
                    f"invalid P-CRAFT grid: {grid_np.shape}, "
                    f"expected {expected}"
                )

            grid = torch.as_tensor(
                np.array(grid_np, dtype=np.float32, copy=True),
                device=device,
            ).unsqueeze(0)

            pooled, fmap = op.pino_encoder.forward_with_map(
                grid,
                preserve_resolution=True,
            )

            scale_globals.append(op.pino_global(pooled))
            scale_maps.append(fmap)

        global_stack = torch.stack(scale_globals, dim=1)

        stats_expanded = statistics[:, None, :].expand(
            -1, len(scale_globals), -1
        )

        scores = op.pino_scale_score(
            torch.cat((global_stack, stats_expanded), dim=-1)
        )

        global_scale_weights = torch.softmax(
            scores.float(), dim=1
        ).to(global_stack.dtype)

        global_context = (
            global_stack * global_scale_weights
        ).sum(dim=1)

        context = (
            op.statistics_context(statistics) + global_context
        ).squeeze(0)

        # Match native P-CRAFT coordinate normalization and axis order.
        origin = torch.as_tensor(
            np.asarray(geometry["origin"], dtype=np.float64),
            device=device,
        )

        span = side * float(geometry["h"])
        normalized = 2 * (xyz - origin) / span - 1

        # grid_sample expects coordinates in (x,y,z) order.
        # Native P-CRAFT stores geometry as (z,y,x).
        sample_coords = torch.stack((
            normalized[:, 2],
            normalized[:, 1],
            normalized[:, 0],
        ), dim=-1)

        pino_map = tuple(scale_maps)
        map_dtype = scale_maps[0].dtype

        # Reuse the exact existing interval sampler, scale gate,
        # interval fusion and residual blocks.
        local_chunks = []
        local_scale_weights = []

        for start in range(0, n, self.chunk_intervals):
            stop = min(start + self.chunk_intervals, n)

            grid = sample_coords[start:stop].to(
                map_dtype
            ).reshape(1, -1, 1, 1, 3)

            local = op._interval_features(
                descriptor[start:stop],
                grid,
                pino_map,
                context,
            )

            local_chunks.append(local)

        local_features = torch.cat(local_chunks, dim=0)

        return {
            "local_features": local_features,
            "global_context": global_context.squeeze(0),
            "global_scale_weights": global_scale_weights.squeeze(0),
            "statistics": statistics.squeeze(0),
            "descriptor": descriptor,
            "normalized_xyz": normalized,
            "interval_mass": mass,
            "geometry_sha256": geometry.get("geometry_sha256", ""),
            "native_resolution": side,
            "n_intervals": n,
        }
