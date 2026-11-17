"""M83 geometry conversion and SQNO geometry-cache management."""

import numpy as np
import torch

from research.NeuralMoments.models.pcraft_ufno.spin_pino.native_pcraft import (
    geometry_from_columns,
    conservative_coarse_grid,
)


def prepare_m83_geometry(raw_geometry, grid_size):
    geometry = geometry_from_columns(raw_geometry)

    geometry["pcraft_grid"] = conservative_coarse_grid(
        geometry,
        grid_size,
    )

    # Preserve the original ten-dimensional material contract.
    fixed = np.asarray(
        geometry["fixed_moments"], dtype=np.float64
    )
    original = np.asarray(
        geometry["original_moments"], dtype=np.float64
    )

    editable = (
        np.asarray(geometry["mass"])[:, None]
        * np.asarray(geometry["moment_basis"])
    ).sum(axis=0)

    if not np.allclose(
        fixed + editable,
        original,
        rtol=1e-8,
        atol=1e-10,
    ):
        raise ValueError(
            "fixed + editable != original moment"
        )

    return geometry


class GeometryCacheRunner:
    def __init__(self, model, grid_size):
        self.model = model
        self.grid_size = int(grid_size)
        self.encode_count = 0

    def encode(self, raw_geometry):
        geometry = prepare_m83_geometry(
            raw_geometry,
            self.grid_size,
        )

        cache = self.model.encode_geometry(geometry)
        self.encode_count += 1
        return cache

    def evaluate(self, cache, queries):
        if not isinstance(queries, torch.Tensor):
            raise TypeError("queries must be a torch.Tensor")

        return self.model.evaluate_queries(
            cache,
            queries,
            with_jacobian=False,
        )
