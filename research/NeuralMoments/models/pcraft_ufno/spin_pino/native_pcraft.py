"""M80 P-CRAFT-dominant operator using the M72 native RLE/query contract.

The coarse 3-D grid is an exact volume aggregation of M72 native RLE columns
when grid_size divides the native side. It is used only for P-CRAFT global
context. Every native interval is retained, sampled against the P-CRAFT
decoder feature map, and included in the mass-weighted interval reduction.
"""
from __future__ import annotations

from collections import OrderedDict
from contextlib import nullcontext
from functools import partial
from dataclasses import asdict, dataclass, replace
from hashlib import sha256
import math
from pathlib import Path
from time import perf_counter
from typing import Mapping
import weakref

import numpy as np
import torch
from torch import Tensor, nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from .pino_backbone import PINOFeatureEncoder
from .geometry_tiles import neighbor_tiles, sample_native_intervals
from ..candidate_projects.quadrann_reproduction.spin_adapter.memory_profile import (
    profile_stage, profile_count, profile_tensors, profile_recompute, profile_adjoint,
)


def checkpoint_stage(name, fn, *args):
    return checkpoint(fn, *args, use_reentrant=False,
                      context_fn=lambda: (nullcontext(), profile_recompute(name)))


SCHEMA = "m80.pcraft-native-query.v5"
SEMANTICS = "pcraft-ufno-native-interval-m72-bounded-odd-or-native-cross-convex-potential-moment-head-ad-jacobian-optional-gno-transolver-multiscale-event-validity.v7"
LEGACY_SCHEMAS = {
    ("m80.pcraft-native-query.v5", "pcraft-ufno-native-interval-m72-bounded-odd-or-convex-potential-moment-head-ad-jacobian-optional-gno-transolver-multiscale-event-validity.v6"),
    ("m80.pcraft-native-query.v5", "pcraft-ufno-native-interval-m72-bounded-odd-or-convex-potential-moment-head-ad-jacobian-optional-gno-transolver-multiscale.v5"),
    ("m80.pcraft-native-query.v1",
     "pcraft-ufno-map-sampled-native-interval-m72-bounded-odd-head-ad-jacobian.v1"),
    ("m80.pcraft-native-query.v2",
     "pcraft-ufno-map-sampled-native-interval-m72-bounded-odd-head-ad-jacobian-optional-gno.v2"),
    ("m80.pcraft-native-query.v3",
     "pcraft-ufno-native-interval-m72-bounded-odd-head-ad-jacobian-optional-gno-transolver.v3"),
    ("m80.pcraft-native-query.v4",
     "pcraft-ufno-native-interval-m72-bounded-odd-head-ad-jacobian-optional-gno-transolver-multiscale.v4"),
}
MOMENT_ORDER = ("mass", "x", "y", "z", "xx", "yy", "zz", "yz", "xz", "xy")
QUERY_ORDER = ("1", "x", "y", "z", "xx", "yy", "zz", "2yz", "2xz", "2xy")
POWERS = (0, 1, 1, 1, 2, 2, 2, 2, 2, 2)
NATIVE_B = (1., 1., 1., 1., 1., 1., 1., 2., 2., 2.)


def native_jacobian_to_monomial(jacobian: Tensor) -> Tensor:
    """Convert d(raw moments)/d(native c) to d(raw moments)/d(B c).

    Native c multiplies (1,x,y,z,xx,yy,zz,2yz,2xz,2xy), whereas the
    reported raw moments contain undoubled cross terms. Therefore the
    convex-potential Hessian is J_c B^-1, not J_c itself.
    """
    if jacobian.ndim < 2 or jacobian.shape[-2:] != (10, 10):
        raise ValueError("native Jacobian must have shape [...,10,10]")
    return jacobian / jacobian.new_tensor(NATIVE_B)

PRESETS = {
    "smoke": dict(grid_size=8, pino_width=2, pino_modes=1, pino_layers=2,
                  interval_width=8, interval_depth=1, chunk_intervals=16),
    "large": dict(grid_size=32, pino_width=16, pino_modes=8, pino_layers=4,
                  interval_width=96, interval_depth=2, chunk_intervals=4096),
    "large_gno": dict(grid_size=32, pino_width=16, pino_modes=8, pino_layers=4,
                      interval_width=96, interval_depth=2, chunk_intervals=2048,
                      gno_enabled=True, gno_neighbors=8),
    "large_transolver": dict(grid_size=32, pino_width=16, pino_modes=8, pino_layers=4,
                             interval_width=96, interval_depth=2, chunk_intervals=2048,
                             transolver_enabled=True, transolver_slices=16, transolver_heads=4),
    "large_gno_transolver": dict(grid_size=32, pino_width=16, pino_modes=8, pino_layers=4,
                                 interval_width=96, interval_depth=2, chunk_intervals=1024,
                                 gno_enabled=True, gno_neighbors=8,
                                 transolver_enabled=True, transolver_slices=16, transolver_heads=4),
    "xl": dict(grid_size=64, pino_width=32, pino_modes=12, pino_layers=6,
              interval_width=192, interval_depth=4, chunk_intervals=2048),
    "xl_gno": dict(grid_size=64, pino_width=32, pino_modes=12, pino_layers=6,
                   interval_width=192, interval_depth=4, chunk_intervals=1024,
                   gno_enabled=True, gno_neighbors=8),
    "xl_transolver": dict(grid_size=64, pino_width=32, pino_modes=12, pino_layers=6,
                          interval_width=192, interval_depth=4, chunk_intervals=1024,
                          transolver_enabled=True, transolver_slices=32, transolver_heads=8),
    "xl_gno_transolver": dict(grid_size=64, pino_width=32, pino_modes=12, pino_layers=6,
                              interval_width=192, interval_depth=4, chunk_intervals=512,
                              gno_enabled=True, gno_neighbors=8,
                              transolver_enabled=True, transolver_slices=32, transolver_heads=8),
    "xxl": dict(grid_size=128, pino_width=48, pino_modes=16, pino_layers=8,
               interval_width=256, interval_depth=6, chunk_intervals=1024),
    "large_convex": dict(grid_size=32, pino_width=16, pino_modes=8, pino_layers=4,
                          interval_width=96, interval_depth=2, chunk_intervals=2048,
                          convex_potential_enabled=True, convex_potential_components=4),
    "xl_convex": dict(grid_size=64, pino_width=32, pino_modes=12, pino_layers=6,
                      interval_width=192, interval_depth=4, chunk_intervals=1024,
                      convex_potential_enabled=True, convex_potential_components=8),
    "large_ms_convex": dict(grid_size=64, pino_width=16, pino_modes=8, pino_layers=4,
                            interval_width=96, interval_depth=2, chunk_intervals=512,
                            multiscale_enabled=True, multiscale_grids=(32,64,128),
                            convex_potential_enabled=True, convex_potential_components=4),
    "large_ms": dict(grid_size=64, pino_width=16, pino_modes=8, pino_layers=4,
                     interval_width=96, interval_depth=2, chunk_intervals=512,
                     multiscale_enabled=True, multiscale_grids=(32,64,128)),
    "large_ms_gno": dict(grid_size=64, pino_width=16, pino_modes=8, pino_layers=4,
                         interval_width=96, interval_depth=2, chunk_intervals=512,
                         multiscale_enabled=True, multiscale_grids=(32,64,128),
                         gno_enabled=True, gno_neighbors=8, gno_topology_filter=True),
    "large_ms_transolver": dict(grid_size=64, pino_width=16, pino_modes=8, pino_layers=4,
                                interval_width=96, interval_depth=2, chunk_intervals=512,
                                multiscale_enabled=True, multiscale_grids=(32,64,128),
                                transolver_enabled=True, transolver_slices=16, transolver_heads=4),
    "large_ms_gno_transolver": dict(grid_size=64, pino_width=16, pino_modes=8, pino_layers=4,
                                    interval_width=96, interval_depth=2, chunk_intervals=512,
                                    multiscale_enabled=True, multiscale_grids=(32,64,128),
                                    gno_enabled=True, gno_neighbors=8, gno_topology_filter=True,
                                    transolver_enabled=True, transolver_slices=16, transolver_heads=4),
    "xl_ms": dict(grid_size=64, pino_width=32, pino_modes=12, pino_layers=6,
                  interval_width=192, interval_depth=4, chunk_intervals=512,
                  multiscale_enabled=True, multiscale_grids=(32,64,128)),
    "xl_ms_convex": dict(grid_size=64, pino_width=32, pino_modes=12, pino_layers=6,
                         interval_width=192, interval_depth=4, chunk_intervals=512,
                         multiscale_enabled=True, multiscale_grids=(32,64,128),
                         convex_potential_enabled=True, convex_potential_components=4),
    "xl_ms_gno": dict(grid_size=64, pino_width=32, pino_modes=12, pino_layers=6,
                      interval_width=192, interval_depth=4, chunk_intervals=512,
                      multiscale_enabled=True, multiscale_grids=(32,64,128),
                      gno_enabled=True, gno_neighbors=8, gno_topology_filter=True),
    "xl_ms_transolver": dict(grid_size=64, pino_width=32, pino_modes=12, pino_layers=6,
                             interval_width=192, interval_depth=4, chunk_intervals=512,
                             multiscale_enabled=True, multiscale_grids=(32,64,128),
                             transolver_enabled=True, transolver_slices=32, transolver_heads=8),
    "xl_ms_gno_transolver": dict(grid_size=64, pino_width=32, pino_modes=12, pino_layers=6,
                                 interval_width=192, interval_depth=4, chunk_intervals=512,
                                 multiscale_enabled=True, multiscale_grids=(32,64,128),
                                 gno_enabled=True, gno_neighbors=8, gno_topology_filter=True,
                                 transolver_enabled=True, transolver_slices=32, transolver_heads=8),
    "xl_ms_gno_transolver_bounded_odd": dict(grid_size=64, pino_width=32, pino_modes=12, pino_layers=6,
                                               interval_width=192, interval_depth=4, chunk_intervals=512,
                                               multiscale_enabled=True, multiscale_grids=(32,64,128),
                                               gno_enabled=True, gno_neighbors=8, gno_topology_filter=True,
                                               transolver_enabled=True, transolver_slices=32, transolver_heads=8,
                                               convex_potential_enabled=False),
    "xl_ms_gno_transolver_convex": dict(grid_size=64, pino_width=32, pino_modes=12, pino_layers=6,
                                          interval_width=192, interval_depth=4, chunk_intervals=512,
                                          multiscale_enabled=True, multiscale_grids=(32,64,128),
                                          gno_enabled=True, gno_neighbors=8, gno_topology_filter=True,
                                          transolver_enabled=True, transolver_slices=32, transolver_heads=8,
                                          convex_potential_enabled=True, convex_potential_components=8),
    "xl_ms_gno_transolver_convex_atan": dict(grid_size=64, pino_width=32, pino_modes=12, pino_layers=6,
                                               interval_width=192, interval_depth=4, chunk_intervals=512,
                                               multiscale_enabled=True, multiscale_grids=(32,64,128),
                                               gno_enabled=True, gno_neighbors=8, gno_topology_filter=True,
                                               transolver_enabled=True, transolver_slices=32, transolver_heads=8,
                                               convex_potential_enabled=True, convex_potential_components=8,
                                               convex_amplitude_mode="atan"),
    "xl_ms_gno_transolver_convex_event": dict(grid_size=64, pino_width=32, pino_modes=12, pino_layers=6,
                                                interval_width=192, interval_depth=4, chunk_intervals=512,
                                                multiscale_enabled=True, multiscale_grids=(32,64,128),
                                                gno_enabled=True, gno_neighbors=8, gno_topology_filter=True,
                                                transolver_enabled=True, transolver_slices=32, transolver_heads=8,
                                                convex_potential_enabled=True, convex_potential_components=8,
                                                event_validity_head_enabled=True),
}


@dataclass(frozen=True)
class NativePCRAFTConfig:
    grid_size: int = 64
    pino_width: int = 32
    pino_modes: int = 12
    pino_layers: int = 6
    interval_width: int = 192
    interval_depth: int = 4
    chunk_intervals: int = 2048
    min_native_resolution: int = 512
    checkpoint_intervals: bool = True
    gno_enabled: bool = False
    gno_neighbors: int = 8
    gno_cache_mib: int = 512
    gno_topology_filter: bool = False
    transolver_enabled: bool = False
    transolver_slices: int = 32
    transolver_heads: int = 4
    transolver_adaptive: bool = False
    transolver_min_slices: int = 4
    transolver_adaptive_temperature: bool = False
    transolver_temperature_min: float = 0.05
    transolver_temperature_max: float = 2.0
    multiscale_enabled: bool = False
    multiscale_grids: tuple[int, ...] = (32, 64, 128)
    convex_potential_enabled: bool = False
    convex_potential_components: int = 4
    convex_amplitude_mode: str = "sigmoid"
    event_validity_head_enabled: bool = False
    uq_enabled: bool = False
    uq_hidden: int = 64
    uq_min_log_std: float = -7.0
    uq_max_log_std: float = 4.0

    def __post_init__(self):
        if self.grid_size < 8 or self.grid_size % 4:
            raise ValueError("grid_size must be >=8 and divisible by four")
        for name in ("pino_width", "pino_modes", "pino_layers", "interval_width",
                     "interval_depth", "chunk_intervals", "min_native_resolution",
                     "gno_neighbors", "gno_cache_mib", "transolver_slices",
                     "transolver_heads"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if not isinstance(self.gno_enabled, bool) or not isinstance(self.transolver_enabled, bool):
            raise ValueError("gno_enabled and transolver_enabled must be bool")
        if not isinstance(self.transolver_adaptive,bool) or not isinstance(self.transolver_adaptive_temperature,bool):
            raise ValueError("adaptive slicing flags must be bool")
        if self.transolver_adaptive_temperature and not self.transolver_adaptive:
            raise ValueError("adaptive temperature requires transolver_adaptive=True")
        if not (0.0 < self.transolver_temperature_min < self.transolver_temperature_max):
            raise ValueError("transolver temperature bounds must be positive and ordered")
        if isinstance(self.transolver_min_slices,bool) or not isinstance(self.transolver_min_slices,int) or not 1<=self.transolver_min_slices<=self.transolver_slices:
            raise ValueError("transolver_min_slices must be in [1, transolver_slices]")
        if not isinstance(self.gno_topology_filter, bool):
            raise ValueError("gno_topology_filter must be bool")
        if not isinstance(self.multiscale_enabled, bool):
            raise ValueError("multiscale_enabled must be bool")
        if not isinstance(self.convex_potential_enabled, bool):
            raise ValueError("convex_potential_enabled must be bool")
        if not isinstance(self.event_validity_head_enabled, bool):
            raise ValueError("event_validity_head_enabled must be bool")
        if not isinstance(self.uq_enabled,bool):
            raise ValueError("uq_enabled must be bool")
        if isinstance(self.uq_hidden,bool) or not isinstance(self.uq_hidden,int) or self.uq_hidden<1:
            raise ValueError("uq_hidden must be a positive integer")
        if not (math.isfinite(self.uq_min_log_std) and math.isfinite(self.uq_max_log_std)
                and -20.0<self.uq_min_log_std<self.uq_max_log_std<10.0):
            raise ValueError("invalid UQ log_std bounds")
        if isinstance(self.convex_potential_components,bool) or not isinstance(self.convex_potential_components,int) or self.convex_potential_components<1:
            raise ValueError("convex_potential_components must be a positive integer")
        if self.convex_amplitude_mode not in ("sigmoid","atan"):
            raise ValueError("convex_amplitude_mode must be 'sigmoid' or 'atan'")
        if (not isinstance(self.multiscale_grids, tuple) or len(self.multiscale_grids) < 2
                or any(isinstance(v,bool) or not isinstance(v,int) or v < 8 or v % 4
                       for v in self.multiscale_grids)
                or tuple(sorted(set(self.multiscale_grids))) != self.multiscale_grids):
            raise ValueError("multiscale_grids must be a sorted tuple of unique >=8 grids divisible by four")
        if self.multiscale_enabled and self.grid_size not in self.multiscale_grids:
            raise ValueError("grid_size must be one of multiscale_grids when multiscale is enabled")
        if self.pino_width < 2 or self.interval_width < 8:
            raise ValueError("pino_width>=2 and interval_width>=8 are required")
        if self.gno_neighbors < 2:
            raise ValueError("gno_neighbors must be >=2")
        if self.interval_width % self.transolver_heads:
            raise ValueError("interval_width must be divisible by transolver_heads")


def make_config(preset="xl", **overrides):
    if preset not in PRESETS:
        raise ValueError(f"unknown preset {preset!r}")
    return NativePCRAFTConfig(**(dict(PRESETS[preset]) | overrides))


def _geometry_hash(columns, origin, h, side, density, fixed):
    digest = sha256()
    for value in (np.asarray(columns, dtype="<i8"), np.asarray(origin, dtype="<f8"),
                  np.asarray([h, density, side], dtype="<f8"), np.asarray(fixed, dtype="<f8")):
        digest.update(value.tobytes())
    return digest.hexdigest()


def geometry_from_columns(values: Mapping[str, object]) -> dict:
    """Port of M72 NativeQuadricOracle.geometry_inputs for M69 RLE NPZ files."""
    columns = np.asarray(values["columns"])
    if columns.ndim != 2 or columns.shape[1] != 4 or columns.dtype.kind not in "iu":
        raise ValueError("columns must be integer [N,4] x,y,lo,hi RLE intervals")
    columns = np.asarray(columns, dtype=np.int64)
    origin = np.asarray(values["origin"], dtype=np.float64)
    h = float(np.asarray(values["h"]))
    side_value = np.asarray(values.get("native_resolution", values.get("side", 0)))
    if side_value.shape != () or side_value.dtype.kind not in "iu":
        raise ValueError("native_resolution/side must be an integer scalar")
    side = int(side_value)
    density = float(np.asarray(values.get("density", 1.0)))
    if origin.shape != (3,) or not np.isfinite(origin).all() or not math.isfinite(h) or h <= 0:
        raise ValueError("invalid origin or voxel width")
    if side < 1 or not math.isfinite(density) or density <= 0:
        raise ValueError("invalid native resolution or density")
    if not len(columns):
        raise ValueError("native RLE geometry has no editable intervals")
    if ((columns[:, :3] < 0).any() or (columns[:, :2] >= side).any()
            or (columns[:, 3] > side).any() or (columns[:, 3] <= columns[:, 2]).any()):
        raise ValueError("RLE interval is outside the native lattice or empty")
    order = np.lexsort((columns[:, 2], columns[:, 1], columns[:, 0]))
    columns = columns[order]
    same_xy = np.all(columns[1:, :2] == columns[:-1, :2], axis=1)
    if np.any(same_xy & (columns[1:, 2] < columns[:-1, 3])):
        raise ValueError("overlapping RLE intervals in one xy column")
    x = origin[0] + (columns[:, 0] + 0.5) * h
    y = origin[1] + (columns[:, 1] + 0.5) * h
    z0, z1 = origin[2] + columns[:, 2]*h, origin[2] + columns[:, 3]*h
    z = (z0+z1)/2
    extent = np.column_stack((np.full(len(columns), h), np.full(len(columns), h), z1-z0))
    mass = density*np.prod(extent, axis=1)
    basis = np.column_stack((np.ones(len(columns)), x, y, z,
        x*x+h*h/12, y*y+h*h/12, (z0*z0+z0*z1+z1*z1)/3, y*z, x*z, x*y))
    fixed = np.asarray(values.get("fixed_moments", np.zeros(10)), dtype=np.float64)
    if fixed.shape != (10,) or not np.isfinite(fixed).all():
        raise ValueError("fixed_moments must be finite [10]")
    design = mass @ basis
    original = np.asarray(values.get("original_moments", fixed+design), dtype=np.float64)
    if original.shape != (10,) or not np.isfinite(original).all():
        raise ValueError("original_moments must be finite [10]")
    tol = max(float(original[0]), float(mass.sum()), 1e-30)*max(side*h, 1e-12)**2*1e-10
    if not np.allclose(fixed+design, original, rtol=1e-8, atol=tol):
        raise ValueError("original moments != fixed + native editable moments")
    return dict(xyz=np.column_stack((x,y,z)), mass=mass, moment_basis=basis,
        cell_extent=extent, fixed_moments=fixed.copy(), original_moments=original.copy(),
        native_resolution=np.asarray(side, dtype=np.int64), density=density,
        geometry_sha256=_geometry_hash(columns,origin,h,side,density,fixed),
        columns=columns.copy(), origin=origin.copy(), h=h, side=side)


def parameter_versions(module):
    # Parameter replacement and Module.to() can preserve the in-place version.
    return tuple((id(p),p._version,p.data_ptr(),p.dtype,p.device,p.requires_grad) for p in module.parameters())


@dataclass(frozen=True)
class NativeFeatureSource:
    """Track the producer of externally supplied learned geometry tensors."""
    owner: weakref.ReferenceType
    versions: tuple
    inference_only: bool

    def validate(self):
        producer=self.owner()
        if producer is None or self.versions!=producer._versions():
            raise ValueError("stale shared P-CRAFT feature payload")


class PreparedNativeGeometry(Mapping):
    """Caller-owned geometry snapshot. Only parameter-independent CPU data are memoized."""

    def __init__(self, values, *, copy_arrays):
        self._data = {}
        for key, value in values.items():
            if isinstance(value, np.ndarray):
                value = value.copy() if copy_arrays else value.view()
                value.flags.writeable = False
            self._data[key] = value
        self._derived = {}

    def __getitem__(self, key):
        return self._data[key]

    def __len__(self):
        return len(self._data)

    def __iter__(self):
        return iter(self._data)

    @classmethod
    def prepare(cls, geometry, *, borrow=False):
        if isinstance(geometry, cls):
            return geometry
        if "xyz" not in geometry:
            values = geometry_from_columns(geometry)
            if "pcraft_grid" in geometry:
                values["pcraft_grid"] = np.array(geometry["pcraft_grid"], copy=True)
            return cls(values, copy_arrays=False)
        return cls(geometry, copy_arrays=not borrow)

    def derived(self, key, build):
        if key not in self._derived:
            self._derived[key] = build()
        return self._derived[key]

    def components(self):
        return self.derived(("components",), lambda: native_interval_components(self["columns"]))

    def grid(self, size):
        return self.derived(("grid", size), lambda: conservative_coarse_grid(self, size))


def load_geometry(path: str | Path) -> dict:
    with np.load(path, allow_pickle=False) as data:
        values = {key: data[key] for key in data.files}
    if "columns" not in values:
        raise ValueError("M80 M72 adapter requires native RLE columns in geometry NPZ")
    return geometry_from_columns(values)


def conservative_coarse_grid(geometry: Mapping[str, object], grid_size: int) -> np.ndarray:
    """Exact RLE voxel-volume aggregation; never replaces the native intervals."""
    columns = np.asarray(geometry["columns"], dtype=np.int64)
    side, h = int(geometry["side"]), float(geometry["h"])
    if side % grid_size:
        raise ValueError("P-CRAFT context grid must divide the native RLE side exactly")
    factor = side//grid_size
    volume = np.zeros((grid_size,grid_size,grid_size), dtype=np.float64)
    zfirst = columns[:,2]//factor
    zlast = (columns[:,3]-1)//factor
    counts = zlast-zfirst+1
    total = int(counts.sum())
    rows = np.repeat(np.arange(len(columns),dtype=np.int64),counts)
    offsets = np.arange(total,dtype=np.int64)-np.repeat(np.cumsum(counts)-counts,counts)
    zb = zfirst[rows]+offsets
    overlap = np.minimum(columns[rows,3],(zb+1)*factor)-np.maximum(columns[rows,2],zb*factor)
    xb, yb = columns[rows,0]//factor, columns[rows,1]//factor
    np.add.at(volume,(xb,yb,zb),overlap.astype(np.float64)*h**3)
    cell = factor*h
    fraction = np.clip(volume/(cell**3),0.,1.)
    from scipy.ndimage import binary_erosion, distance_transform_edt
    occupied = fraction>0
    if not occupied.any():
        raise ValueError("coarse context grid has no editable material")
    signed = (-np.ones_like(fraction) if occupied.all()
              else distance_transform_edt(~occupied)-distance_transform_edt(occupied))
    boundary = occupied & ~binary_erosion(occupied,structure=np.ones((3,3,3),dtype=bool),border_value=0)
    return np.stack((fraction,occupied.astype(float),np.tanh(signed/2),boundary.astype(float))).astype(np.float32)


def _absolute_basis(xyz: Tensor, extent: Tensor) -> Tensor:
    lo,hi=xyz-extent/2,xyz+extent/2
    integral=(hi*hi.abs()-lo*lo.abs())/(2*extent.clamp_min(torch.finfo(extent.dtype).tiny))
    absolute=torch.where(extent>0,integral,xyz.abs())
    second=xyz.square()+extent.square()/12
    x,y,z=absolute.unbind(-1)
    return torch.stack((torch.ones_like(x),x,y,z,second[:,0],second[:,1],second[:,2],y*z,x*z,x*y),1)


def _quadric_bounds(bounds: Tensor, q: Tensor):
    lo,hi=bounds[0],bounds[1]
    sqlo=torch.where((lo<=0)&(hi>=0),torch.zeros_like(lo),torch.minimum(lo.square(),hi.square()))
    sqhi=torch.maximum(lo.square(),hi.square())
    low=[q.new_tensor(1.),*lo.unbind(),*sqlo.unbind()]
    high=[q.new_tensor(1.),*hi.unbind(),*sqhi.unbind()]
    for i,j in ((1,2),(0,2),(0,1)):
        corners=torch.stack((lo[i]*lo[j],lo[i]*hi[j],hi[i]*lo[j],hi[i]*hi[j]))
        low.append(2*corners.min()); high.append(2*corners.max())
    a,b=torch.stack(low),torch.stack(high)
    return torch.minimum(q*a,q*b).sum(-1),torch.maximum(q*a,q*b).sum(-1)


@dataclass(frozen=True)
class EncodedNativeGeometry:
    latent: Tensor
    statistics: Tensor
    fixed_moments: Tensor
    design_moments: Tensor
    absolute_moments: Tensor
    moment_scales: Tensor
    coordinate_scales: Tensor
    normalized_bounds: Tensor
    geometry_sha256: str
    native_resolution: int
    native_intervals: int
    owner: int
    versions: tuple
    interval_features: Tensor | None = None
    global_context: Tensor | None = None
    interval_mass: Tensor | None = None
    descriptor: Tensor | None = None
    normalized_xyz: Tensor | None = None
    global_scale_weights: Tensor | None = None
    scale_features: Tensor | None = None
    scale_ids: Tensor | None = None

    def detached(self):
        names=("latent","statistics","fixed_moments","design_moments","absolute_moments",
               "moment_scales","coordinate_scales","normalized_bounds")
        optional=("interval_features","global_context","interval_mass","descriptor",
                  "normalized_xyz","global_scale_weights","scale_features","scale_ids")
        values={name:getattr(self,name).detach() for name in names}
        values.update({name:getattr(self,name).detach() for name in optional
                       if getattr(self,name) is not None})
        return replace(self,**values)


class IntervalResidual(nn.Module):
    def __init__(self,width):
        super().__init__()
        self.norm=nn.LayerNorm(width)
        self.ffn=nn.Sequential(nn.Linear(width,4*width),nn.GELU(),nn.Linear(4*width,width))
    def forward(self,x):
        return x+self.ffn(self.norm(x))


class PhysicsStateSlicing(nn.Module):
    """Deterministic, mass-aware adaptation of Transolver++ physical slicing.

    The upstream Transolver++ mechanism projects native points into multiple
    heads, learns per-point slice assignments, pools slice tokens, applies
    attention over the small slice axis, and deslices the updated states back
    to points. This Spin-specific version weights slice pooling by native
    material mass and uses deterministic soft assignments (rather than
    stochastic Gumbel sampling) to keep repeated geometry queries reproducible.

    Complexity is O(N * H * S * D + H * S^2 * D), not O(N^2).
    """
    def __init__(self,width:int,n_slices:int,n_heads:int):
        super().__init__()
        if width % n_heads:
            raise ValueError("state width must be divisible by attention heads")
        self.width=int(width)
        self.n_slices=int(n_slices)
        self.n_heads=int(n_heads)
        self.dim_head=width//n_heads
        self.norm=nn.LayerNorm(width)
        self.in_project_x=nn.Linear(width,width)
        self.in_project_slice=nn.Linear(self.dim_head,n_slices)
        nn.init.orthogonal_(self.in_project_slice.weight)
        nn.init.zeros_(self.in_project_slice.bias)
        self.proj_temperature=nn.Sequential(
            nn.Linear(self.dim_head,n_slices),nn.GELU(),
            nn.Linear(n_slices,1),nn.GELU()
        )
        self.temperature_bias=nn.Parameter(torch.full((1,n_heads,1),0.5))
        self.to_q=nn.Linear(self.dim_head,self.dim_head,bias=False)
        self.to_k=nn.Linear(self.dim_head,self.dim_head,bias=False)
        self.to_v=nn.Linear(self.dim_head,self.dim_head,bias=False)
        self.to_out=nn.Linear(width,width)
        self.attn_dropout=nn.Dropout(0.0)
        self.ffn_norm=nn.LayerNorm(width)
        self.ffn=nn.Sequential(nn.Linear(width,4*width),nn.GELU(),nn.Linear(4*width,width))
        self.ffn_dropout=nn.Dropout(0.0)

    def forward(self,features:Tensor,mass_weights:Tensor,*,return_details:bool=False,
                adaptive:bool=False,min_slices:int|None=None,
                adaptive_temperature:bool=False,temperature_min:float=0.05,
                temperature_max:float=2.0,point_chunk_size:int=16384):
        if features.ndim!=2 or features.shape[1]!=self.width:
            raise ValueError("features must have shape [N,width]")
        if mass_weights.ndim!=1 or len(features)!=len(mass_weights):
            raise ValueError("mass_weights must have shape [N]")
        if point_chunk_size<1:
            raise ValueError("point_chunk_size must be positive")
        if not torch.isfinite(mass_weights).all():
            raise ValueError("nonfinite physical state masses")
        if (mass_weights<0).any() or float(mass_weights.sum().detach())<=0:
            raise ValueError("mass weights must be nonnegative with positive sum")

        n=len(features)
        mass=mass_weights.float()
        mass=mass/mass.sum().clamp_min(1e-30)
        feature_tiles=features.split(point_chunk_size)
        mass_tiles=mass.split(point_chunk_size)
        active_slices=self.n_slices
        feature_variation=features.new_zeros(())
        mass_effective_support=features.new_tensor(float(n))
        complexity=features.new_tensor(1.0)
        if adaptive or adaptive_temperature:
            # Stream global weighted statistics; never materialize a second
            # float64 copy of the million-interval feature matrix.
            mass64=mass_weights.to(torch.float64).clamp_min(0)
            total_mass=mass64.sum().clamp_min(torch.finfo(torch.float64).tiny)
            mass_effective_support=total_mass.square() / mass64.square().sum().clamp_min(1e-15)
            weighted_sum=features.new_zeros((self.width,),dtype=torch.float64)
            weighted_energy=features.new_zeros((),dtype=torch.float64)
            weights64=(mass64/total_mass).split(point_chunk_size)
            def mean_and_energy(f,w):
                f=f.double()
                return (f*w[:,None]).sum(0),(f.square().mean(-1)*w).sum()
            def variance(f,w,mean):
                return ((f.double()-mean).square().mean(-1)*w).sum()
            def statistic(fn,*args):
                if self.training and torch.is_grad_enabled():
                    return checkpoint_stage("transolver_statistics_recompute",fn,*args)
                return fn(*args)
            for f,w in zip(feature_tiles,weights64):
                chunk_mean,chunk_energy=statistic(mean_and_energy,f,w)
                weighted_sum=weighted_sum+chunk_mean
                weighted_energy=weighted_energy+chunk_energy
            weighted_variance=features.new_zeros((),dtype=torch.float64)
            for f,w in zip(feature_tiles,weights64):
                weighted_variance=weighted_variance+statistic(variance,f,w,weighted_sum)
            feature_variation=(weighted_variance/weighted_energy.clamp_min(1e-12)).clamp(0,1)
            mass_coverage=(mass_effective_support/max(1,n)).clamp(0,1)
            complexity=(0.5*feature_variation+0.5*mass_coverage).clamp(0,1)
            minimum=int(min_slices or max(1,self.n_slices//4))
            if not 1<=minimum<=self.n_slices:
                raise ValueError("min_slices must be in [1,n_slices]")
            if adaptive:
                active_slices=minimum+int(torch.round(complexity*(self.n_slices-minimum)).detach().cpu())

        def project_and_assign(f:Tensor,m:Tensor):
            x_mid=self.in_project_x(self.norm(f)).reshape(len(f),self.n_heads,self.dim_head)
            logits=self.in_project_slice(x_mid)[...,:active_slices]
            temperature=(self.proj_temperature(x_mid)+self.temperature_bias).clamp_min(0.01)
            if adaptive_temperature:
                if not (0.0<temperature_min<temperature_max):
                    raise ValueError("adaptive temperature bounds must be positive and ordered")
                temperature=(temperature*(1.5-complexity)).clamp(
                    min=temperature_min,max=temperature_max
                )
            assignment=torch.softmax(logits.float()/temperature.float(),dim=-1)
            if active_slices<self.n_slices:
                assignment=F.pad(assignment,(0,self.n_slices-active_slices))
            return x_mid,assignment,temperature

        slice_mass=features.new_zeros((self.n_heads,self.n_slices),dtype=torch.float32)
        slice_numerator=features.new_zeros((self.n_heads,self.n_slices,self.dim_head),dtype=torch.float32)
        entropy_sum=features.new_zeros((),dtype=torch.float64)
        temperature_sum=features.new_zeros((),dtype=torch.float64)
        temperature_min_value=float("inf")
        temperature_max_value=float("-inf")
        row_error=features.new_zeros(())
        for f,m in zip(feature_tiles,mass_tiles):
            def pool_fn(f_chunk,m_chunk):
                profile_count(transolver_pool_chunks=1)
                x_mid,assignment,temp=project_and_assign(f_chunk,m_chunk)
                weighted=assignment*m_chunk[:,None,None]
                sm=weighted.sum(0)
                sn=torch.einsum("nhd,nhs->hsd",x_mid.float(),weighted)
                ent=-(assignment*assignment.clamp_min(1e-30).log()).sum()
                tsum=temp.to(torch.float64).sum()
                return sm,sn,ent,tsum,temp.detach().amin(),temp.detach().amax(),(assignment.sum(-1)-1).abs().max()
            if self.training and torch.is_grad_enabled():
                sm,sn,ent,tsum,tmin,tmax,rerr=checkpoint_stage(
                    "transolver_pool_recompute",pool_fn,f,m
                )
            else:
                sm,sn,ent,tsum,tmin,tmax,rerr=pool_fn(f,m)
            slice_mass=slice_mass+sm
            slice_numerator=slice_numerator+sn
            entropy_sum=entropy_sum+ent.to(torch.float64)
            temperature_sum=temperature_sum+tsum
            temperature_min_value=min(temperature_min_value,float(tmin))
            temperature_max_value=max(temperature_max_value,float(tmax))
            row_error=torch.maximum(row_error,rerr)

        slice_tokens=slice_numerator/(slice_mass.unsqueeze(-1)+1e-8)
        q=self.to_q(slice_tokens).unsqueeze(0)
        k=self.to_k(slice_tokens).unsqueeze(0)
        v=self.to_v(slice_tokens).unsqueeze(0)
        # Only the global slice axis is quadratic. Math SDPA also supports
        # double backward through the slice interaction on CPU and CUDA.
        with torch.nn.attention.sdpa_kernel(torch.nn.attention.SDPBackend.MATH):
            attended=F.scaled_dot_product_attention(q,k,v).squeeze(0)

        output_chunks=[]
        assignments=[]
        for f,m in zip(feature_tiles,mass_tiles):
            def deslice_fn(f_chunk,m_chunk):
                profile_count(transolver_deslice_chunks=1)
                x_mid,assignment,_=project_and_assign(f_chunk,m_chunk)
                point_update=torch.einsum("hsd,nhs->nhd",attended,assignment).reshape(len(f_chunk),self.width)
                x=f_chunk+self.attn_dropout(self.to_out(point_update).to(f_chunk.dtype))
                updated=x+self.ffn_dropout(self.ffn(self.ffn_norm(x)).to(x.dtype))
                return (updated,assignment) if return_details else updated
            if self.training and torch.is_grad_enabled():
                updated=checkpoint_stage("transolver_deslice_recompute",deslice_fn,f,m)
            else:
                updated=deslice_fn(f,m)
            if return_details:
                updated,assignment=updated
                assignments.append(assignment)
            output_chunks.append(updated)
        output=torch.cat(output_chunks,dim=0)

        mean_usage=slice_mass.mean(dim=0)
        entropy=-(mean_usage*mean_usage.clamp_min(1e-30).log()).sum()
        head_entropy=-(slice_mass*slice_mass.clamp_min(1e-30).log()).sum(dim=-1)
        head_effective=torch.exp(head_entropy)
        occupancy_threshold=1e-6
        near_empty_threshold=1e-4
        per_head_occupied=(slice_mass>occupancy_threshold).sum(dim=-1)
        per_head_near_empty=((slice_mass>0)&(slice_mass<=near_empty_threshold)).sum(dim=-1)
        metrics={
            "transolver_assignment_row_error":float(row_error.detach()),
            "transolver_slice_mass_sum_error":float((slice_mass.sum(-1)-1).abs().max().detach()),
            "transolver_effective_slices":float(torch.exp(entropy).detach()),
            "transolver_per_head_effective_slices_mean":float(head_effective.mean().detach()),
            "transolver_per_head_effective_slices_min":float(head_effective.min().detach()),
            "transolver_per_head_effective_slices_max":float(head_effective.max().detach()),
            "transolver_per_head_occupied_slices_min":float(per_head_occupied.min().detach()),
            "transolver_per_head_occupied_slices_max":float(per_head_occupied.max().detach()),
            "transolver_per_head_near_empty_slices_mean":float(per_head_near_empty.float().mean().detach()),
            "transolver_active_slice_utilization":float((head_effective.mean()/max(1,active_slices)).detach()),
            "transolver_assignment_entropy":float((entropy_sum/(n*self.n_heads)).detach()),
            "transolver_temperature_mean":float((temperature_sum/(n*self.n_heads)).detach()),
            "transolver_temperature_min":temperature_min_value,
            "transolver_temperature_max":temperature_max_value,
            "transolver_adaptive_enabled":float(bool(adaptive)),
            "transolver_adaptive_temperature_enabled":float(bool(adaptive_temperature)),
            "transolver_temperature_bound_min":float(temperature_min),
            "transolver_temperature_bound_max":float(temperature_max),
            "transolver_active_slices":float(active_slices),
            "transolver_feature_variation":float(feature_variation.detach()),
            "transolver_mass_effective_support":float(mass_effective_support.detach()),
            "transolver_complexity_score":float(complexity.detach()) if adaptive else 1.0,
            "transolver_point_chunk_size":float(point_chunk_size),
        }
        assignment=torch.cat(assignments) if return_details else None
        profile_adjoint("transolver",outputs=(output,assignment,slice_mass if return_details else None),
                        inputs=(features,mass_weights),parameters=self.parameters())
        if return_details:
            return output,metrics,assignment,slice_mass
        return output,metrics



class ConvexMomentPotentialHead(nn.Module):
    """Smooth even convex 1-homogeneous potential for editable-domain moments.

    Psi(q_tilde) = sum_k a_k * sqrt(q_tilde^T C_k q_tilde), where each C_k
    is an SPD correlation matrix (unit diagonal) and a_k >= 0. The response
    is r_D(c) = 0.5 * M_D + D * grad(Psi(D B c)), where D contains
    coordinate_scales and B doubles the three native cross coefficients.
    Thus J_c B^-1 = D Hessian(Psi) D is symmetric PSD in smooth nonzero
    query directions. Raw J_c need not be symmetric. Positive-scale
    invariance and the M72 complement relation are retained. A geometry-
    dependent amplitude bound keeps deviations within the raw-moment envelope.
    """
    def __init__(self,width:int,components:int=4,min_diag:float=1e-3,amplitude_mode:str="sigmoid"):
        super().__init__()
        if components<1 or min_diag<=0:
            raise ValueError("components and min_diag must be positive")
        self.components=int(components)
        if amplitude_mode not in ("sigmoid","atan"):
            raise ValueError("amplitude_mode must be 'sigmoid' or 'atan'")
        self.min_diag=float(min_diag)
        self.amplitude_mode=amplitude_mode
        self.parameter_net=nn.Sequential(
            nn.LayerNorm(width),nn.Linear(width,2*width),nn.GELU(),
            nn.Linear(2*width,self.components*56)
        )
        self.total_gate=nn.Linear(width,1)
        nn.init.constant_(self.total_gate.bias,-1.0)

    def amplitude_parameters(self,latent:Tensor):
        logits=self.total_gate(latent).reshape(()).double()
        fraction=torch.sigmoid(logits) if self.amplitude_mode=="sigmoid" else 0.5+torch.atan(logits)/math.pi
        return logits,fraction

    def forward(self,latent:Tensor,q_tilde:Tensor,amplitude_bound:Tensor):
        if q_tilde.ndim!=2 or q_tilde.shape[-1]!=10:
            raise ValueError("q_tilde must have shape [Q,10]")
        raw=self.parameter_net(latent).reshape(self.components,56).double()
        amp_logits=raw[:,0]
        tri=raw[:,1:]
        rows,cols=torch.tril_indices(10,10,device=q_tilde.device)
        lower=torch.zeros((self.components,10,10),dtype=torch.float64,device=q_tilde.device)
        lower[:,rows,cols]=tri
        diag_index=torch.arange(10,device=q_tilde.device)
        diagonal=F.softplus(lower[:,diag_index,diag_index])+self.min_diag
        lower[:,diag_index,diag_index]=diagonal
        gram=lower@lower.transpose(-1,-2)
        gram_diag=torch.diagonal(gram,dim1=-2,dim2=-1).clamp_min(1e-24).sqrt()
        correlation=gram/(gram_diag.unsqueeze(-1)*gram_diag.unsqueeze(-2))
        gate_logits,fraction=self.amplitude_parameters(latent)
        total_amplitude=amplitude_bound.to(torch.float64)*fraction
        amplitudes=torch.softmax(amp_logits,dim=0)*total_amplitude
        quadratic=torch.einsum("qi,kij,qj->qk",q_tilde,correlation,q_tilde)
        norms=torch.sqrt(quadratic.clamp_min(torch.finfo(torch.float64).tiny))
        potential=(norms*amplitudes.unsqueeze(0)).sum(-1)
        return potential,fraction,gate_logits

def native_interval_components(columns: np.ndarray) -> np.ndarray:
    """Label 6-connected components of M72 x/y/z RLE intervals.

    RLE rows are (x, y, z_lo, z_hi), with half-open z ranges. Intervals
    connect vertically when they touch in the same x/y column, and connect
    across x/y faces only when their z ranges overlap. Edge/corner-only
    contacts do not connect. This prevents Euclidean kNN from linking nearby
    but disconnected thin-wall layers.
    """
    columns=np.asarray(columns)
    if columns.ndim!=2 or columns.shape[1]!=4 or columns.dtype.kind not in "iu" or len(columns)==0:
        raise ValueError("columns must be nonempty integer [N,4] RLE intervals")
    rows=np.asarray(columns,dtype=np.int64)
    if (rows[:,3]<=rows[:,2]).any():
        raise ValueError("RLE intervals must have positive z extent")
    groups={}
    for i,(x,y,z0,z1) in enumerate(rows):
        groups.setdefault((int(x),int(y)),[]).append((int(z0),int(z1),i))
    parent=np.arange(len(rows),dtype=np.int64)
    rank=np.zeros(len(rows),dtype=np.int8)
    def find(a):
        while parent[a]!=a:
            parent[a]=parent[parent[a]]
            a=int(parent[a])
        return a
    def union(a,b):
        ra,rb=find(a),find(b)
        if ra==rb:return
        if rank[ra]<rank[rb]:ra,rb=rb,ra
        parent[rb]=ra
        if rank[ra]==rank[rb]:rank[ra]+=1
    ordered={}
    for key,items in groups.items():
        items=sorted(items)
        for left,right in zip(items,items[1:]):
            if left[1]>right[0]:
                raise ValueError("overlapping RLE intervals in one x/y column")
            if left[1]==right[0]:
                union(left[2],right[2])
        ordered[key]=items
    for (x,y),items in ordered.items():
        for neighbor_key in ((x+1,y),(x,y+1)):
            other=ordered.get(neighbor_key)
            if not other:continue
            i=j=0
            while i<len(items) and j<len(other):
                a0,a1,ai=items[i];b0,b1,bi=other[j]
                if max(a0,b0)<min(a1,b1):
                    union(ai,bi)
                if a1<=b1:i+=1
                if b1<=a1:j+=1
    roots=[find(i) for i in range(len(rows))]
    remap={root:index for index,root in enumerate(sorted(set(roots)))}
    return np.asarray([remap[root] for root in roots],dtype=np.int32)


def build_local_neighbor_graph(
    xyz: np.ndarray, neighbors: int = 8, component_ids: np.ndarray | None = None
) -> np.ndarray:
    """Build deterministic self-inclusive kNN, optionally within 6-connected components.

    The graph is geometry-only, not a dense N-by-N matrix. If component_ids
    are supplied, no edge crosses a disconnected native material component.
    Output is int32 [N,K] to limit memory.
    """
    xyz=np.asarray(xyz,dtype=np.float64)
    if xyz.ndim!=2 or xyz.shape[1]!=3 or len(xyz)==0 or not np.isfinite(xyz).all():
        raise ValueError("xyz must be finite [N,3]")
    if neighbors<2:
        raise ValueError("neighbors must be >=2")
    n=len(xyz)
    if component_ids is None:
        groups=[np.arange(n,dtype=np.int64)]
    else:
        labels=np.asarray(component_ids)
        if labels.shape!=(n,) or labels.dtype.kind not in "iu":
            raise ValueError("component_ids must be integer [N]")
        groups=[np.flatnonzero(labels==label) for label in np.unique(labels)]
    from scipy.spatial import cKDTree
    result=np.empty((n,neighbors),dtype=np.int32)
    for indices_global in groups:
        points=xyz[indices_global]
        k=min(len(indices_global),int(neighbors))
        if k==1:
            result[indices_global,:]=indices_global[:,None].astype(np.int32)
            continue
        tree=cKDTree(points)
        for start in range(0,len(points),32768):
            stop=min(start+32768,len(points))
            _,indices=tree.query(points[start:stop],k=k,workers=1)
            own=np.arange(start,stop,dtype=np.int64)
            for row in np.flatnonzero(indices[:,0]!=own):
                original=indices[row]
                indices[row]=np.concatenate(([own[row]],original[original!=own[row]]))[:k]
            target=indices_global[start:stop]
            result[target,:k]=indices_global[indices]
            if k<neighbors:
                result[target,k:]=target[:,None]
    return result


class PcraftNativeMomentOperator(nn.Module):
    """P-CRAFT spatial operator + M72 native interval response head."""
    def __init__(self,config: NativePCRAFTConfig|None=None, *, encoder_only=False):
        super().__init__()
        self.config=config or make_config("xl")
        self.encoder_only=bool(encoder_only)
        if self.encoder_only and (self.config.uq_enabled or self.config.event_validity_head_enabled):
            raise ValueError("encoder-only P-CRAFT cannot construct auxiliary prediction heads")
        c=self.config; w=c.interval_width
        self.pino_encoder=PINOFeatureEncoder(width=c.pino_width,modes=c.pino_modes,
            n_layers=c.pino_layers,grid_size=c.grid_size,use_gradient_checkpoint=True)
        if c.checkpoint_intervals:
            for block in (*self.pino_encoder.enc1,*self.pino_encoder.dec1):
                block.use_checkpoint=True
        self.pino_global=nn.Sequential(nn.Linear(self.pino_encoder.feature_dim,w),nn.GELU(),nn.Linear(w,w))
        self.pino_local=nn.Linear(c.pino_width,w,bias=False)
        if c.multiscale_enabled:
            self.pino_scale_score=nn.Sequential(nn.Linear(w+36,w),nn.GELU(),nn.Linear(w,1))
            self.local_scale_gate=nn.Sequential(nn.Linear(13+2*w,w),nn.GELU(),nn.Linear(w,1))
        self.statistics_context=nn.Sequential(nn.Linear(36,w),nn.SiLU(),nn.Linear(w,w))
        self.interval_embedding=nn.Sequential(nn.Linear(13,w),nn.GELU(),nn.Linear(w,w))
        self.interval_fusion=nn.Sequential(nn.LayerNorm(w),nn.Linear(w,2*w),nn.GELU(),nn.Linear(2*w,w))
        self.interval_blocks=nn.ModuleList(IntervalResidual(w) for _ in range(c.interval_depth))
        if c.gno_enabled:
            hidden=max(16,w//2)
            self.gno_kernel=nn.Sequential(nn.Linear(7,hidden),nn.SiLU(),nn.Linear(hidden,1))
            self.gno_value=nn.Linear(w,w,bias=False)
            self.gno_update=nn.Sequential(nn.LayerNorm(2*w),nn.Linear(2*w,2*w),
                                          nn.GELU(),nn.Linear(2*w,w))
            nn.init.normal_(self.gno_update[-1].weight,std=1e-3)
            nn.init.zeros_(self.gno_update[-1].bias)
        if c.transolver_enabled:
            self.transolver=PhysicsStateSlicing(w,c.transolver_slices,c.transolver_heads)
        if self.encoder_only:
            pass
        elif c.convex_potential_enabled:
            self.convex_potential_head=ConvexMomentPotentialHead(w,c.convex_potential_components,
                amplitude_mode=c.convex_amplitude_mode)
        else:
            self.query_head=nn.Sequential(nn.Linear(w+36+10,2*w),nn.SiLU(),
                                          nn.Linear(2*w,w),nn.SiLU(),nn.Linear(w,10))
            nn.init.normal_(self.query_head[-1].weight,std=.01)
            nn.init.zeros_(self.query_head[-1].bias)
        if c.uq_enabled:
            self.uq_head=nn.Sequential(
                nn.LayerNorm(w+36+10),nn.Linear(w+36+10,c.uq_hidden),nn.GELU(),
                nn.Linear(c.uq_hidden,110)
            )
            nn.init.zeros_(self.uq_head[-1].weight)
            nn.init.zeros_(self.uq_head[-1].bias)
        if c.event_validity_head_enabled:
            self.event_validity_head=nn.Sequential(
                nn.LayerNorm(w+36+10),nn.Linear(w+36+10,w),nn.GELU(),
                nn.Linear(w,max(8,w//2)),nn.GELU(),nn.Linear(max(8,w//2),1)
            )
        self.register_buffer("native_B",torch.tensor(NATIVE_B,dtype=torch.float64))
        self._neighbor_cache=OrderedDict()
        self._neighbor_cache_bytes=0
        self.counters=dict(encode_calls=0,intervals=0,queries=0,jacobians=0,exact_calls=0,
                           gno_graph_builds=0,gno_graph_cache_hits=0,gno_graph_build_seconds=0.0,
                           gno_edges=0,gno_component_count=0,transolver_calls=0,transolver_effective_slices=0.0,
                           transolver_assignment_entropy=0.0,
                           transolver_assignment_row_error=0.0,
                           transolver_slice_mass_sum_error=0.0,
                           transolver_temperature_mean=0.0,
                           transolver_temperature_min=0.0,
                           transolver_temperature_max=0.0,
                           multiscale_calls=0,
                           multiscale_global_weights=[],
                           multiscale_local_weights=[],
                           convex_potential_calls=0,
                           convex_potential_scale_fraction=0.0,
                           convex_potential_gate_logit=0.0,
                           convex_potential_amplitude_bound=0.0,
                           convex_potential_effective_amplitude=0.0)

    def geometry_parameters(self):
        for name,layer in self.named_children():
            if not name.endswith("_head"):
                yield from layer.parameters()

    def _versions(self):
        return parameter_versions(self)

    def _get_neighbor_graph(self,geometry_hash,xyz,device,geometry):
        key=(geometry_hash,self.config.gno_neighbors,str(device))
        if key in self._neighbor_cache:
            graph,size=self._neighbor_cache.pop(key)
            self._neighbor_cache[key]=(graph,size)
            self.counters["gno_graph_cache_hits"]+=1
            return graph
        graph_start=perf_counter()
        component_ids=None
        if self.config.gno_topology_filter:
            columns=geometry.get("columns")
            if columns is None:
                raise ValueError("topology-filtered GNO requires native RLE columns")
            component_ids=geometry.components()
            self.counters["gno_component_count"]=int(len(np.unique(component_ids)))
        graph=geometry.derived(("knn",self.config.gno_neighbors,xyz.dtype.str,self.config.gno_topology_filter),
            lambda: build_local_neighbor_graph(xyz,self.config.gno_neighbors,component_ids=component_ids))
        size=graph.nbytes
        budget=self.config.gno_cache_mib*1024**2
        if size<=budget:
            while self._neighbor_cache and self._neighbor_cache_bytes+size>budget:
                _,(_,old_size)=self._neighbor_cache.popitem(last=False)
                self._neighbor_cache_bytes-=old_size
            self._neighbor_cache[key]=(graph,size)
            self._neighbor_cache_bytes+=size
        self.counters["gno_graph_builds"]+=1
        self.counters["gno_graph_build_seconds"]+=perf_counter()-graph_start
        self.counters["gno_edges"]+=int(graph.size)
        return graph

    def _interval_features(self,descriptor,grid,pino_map,context):
        # Also used by the stand-alone v1.1 adapter with pre-built sampling grids.
        maps=pino_map if isinstance(pino_map,(tuple,list)) else (pino_map,)
        samples=tuple(sample_native_intervals(fmap,grid.reshape(-1,3)) for fmap in maps)
        return self._features_from_samples(descriptor,samples,context)

    def _features_from_samples(self,descriptor,samples,context):
        profile_count(interval_feature_chunks=1)
        if self.config.multiscale_enabled:
            projected=[]
            scores=[]
            ctx=context.expand(len(descriptor),-1)
            for sampled in samples:
                local=self.pino_local(sampled)
                projected.append(local)
                scores.append(self.local_scale_gate(torch.cat((descriptor,ctx,local),dim=-1)))
            score=torch.cat(scores,dim=-1)
            gate=torch.softmax(score.float(),dim=-1).to(projected[0].dtype)
            self.counters["multiscale_local_weights"]=[float(v) for v in gate.detach().float().mean(0).cpu()]
            stacked=torch.stack(projected,dim=1)
            local_fused=(stacked*gate.unsqueeze(-1)).sum(dim=1)
            h=self.interval_embedding(descriptor)+local_fused
        else:
            h=self.interval_embedding(descriptor)+self.pino_local(samples[0])
        h=self.interval_fusion(h+context)
        for block in self.interval_blocks:
            h=block(h)
        return h

    def _chunk_pool(self,descriptor,weights,samples,context):
        h=self._features_from_samples(descriptor,samples,context)
        return (h*weights[:,None]).sum(0)

    def _gno_update_chunk(self,center_features,neighbor_features,edge_features,neighbor_weights):
        kernel=F.softplus(self.gno_kernel(edge_features.to(center_features.dtype)).squeeze(-1))
        kernel=kernel*neighbor_weights.to(kernel.dtype).clamp_min(1e-12)
        alpha=kernel/(kernel.sum(-1,keepdim=True)+1e-12)
        local_integral=(alpha.unsqueeze(-1)*self.gno_value(neighbor_features)).sum(1)
        return center_features+self.gno_update(torch.cat((center_features,local_integral),dim=-1))

    def encode(self,geometry: Mapping[str,object],chunk: int|None=None,*,retain_interval_features=False):
        c=self.config; w=c.interval_width; chunk=c.chunk_intervals if chunk is None else chunk
        if chunk<1: raise ValueError("chunk must be positive")
        geometry=PreparedNativeGeometry.prepare(geometry,borrow=True)
        xyz_np=np.asarray(geometry["xyz"],dtype=np.float64)
        mass_np=np.asarray(geometry["mass"],dtype=np.float64)
        extent_np=np.asarray(geometry["cell_extent"],dtype=np.float64)
        basis_np=np.asarray(geometry["moment_basis"],dtype=np.float64)
        if xyz_np.ndim!=2 or xyz_np.shape[1]!=3 or len(xyz_np)==0: raise ValueError("invalid native interval xyz")
        if mass_np.shape!=(len(xyz_np),) or extent_np.shape!=(len(xyz_np),3) or basis_np.shape!=(len(xyz_np),10):
            raise ValueError("M72 interval arrays have inconsistent shapes")
        if not np.isfinite(xyz_np).all() or not np.isfinite(mass_np).all() or not np.isfinite(extent_np).all() or not np.isfinite(basis_np).all() or (mass_np<=0).any():
            raise ValueError("nonfinite or nonpositive native interval data")
        side=int(np.asarray(geometry["native_resolution"]))
        if side<c.min_native_resolution: raise ValueError("native geometry resolution below configured minimum")
        p=next(self.parameters()); device=p.device
        xyz=torch.as_tensor(xyz_np.copy(),dtype=torch.float64,device=device)
        mass=torch.as_tensor(mass_np.copy(),dtype=torch.float64,device=device)
        extent=torch.as_tensor(extent_np.copy(),dtype=torch.float64,device=device)
        basis=torch.as_tensor(basis_np.copy(),dtype=torch.float64,device=device)
        fixed=torch.as_tensor(np.asarray(geometry["fixed_moments"],dtype=np.float64).copy(),dtype=torch.float64,device=device)
        design=mass@basis
        original=torch.as_tensor(np.asarray(geometry["original_moments"],dtype=np.float64).copy(),dtype=torch.float64,device=device)
        if fixed.shape!=(10,) or original.shape!=(10,) or not torch.isfinite(fixed).all() or not torch.isfinite(original).all():
            raise ValueError("fixed/original moments must be finite [10]")
        lo,hi=(xyz-extent/2).amin(0),(xyz+extent/2).amax(0)
        length=(hi-lo).max().clamp_min(torch.finfo(torch.float64).eps)
        coord=length**torch.tensor(POWERS,dtype=torch.float64,device=device)
        scale=(fixed[0]+mass.sum())*coord
        if not torch.allclose(fixed+design,original,rtol=1e-8,atol=float(scale.max())*1e-10):
            raise ValueError("M72 original moments do not equal fixed + editable moments")
        absolute=mass@_absolute_basis(xyz,extent)
        if (absolute+1e-10*scale<design.abs()).any(): raise ValueError("absolute moment bound violated")
        stats=torch.cat((original/scale,fixed/scale,design/scale,(lo+hi)/(2*length),(hi-lo)/length)).to(p.dtype)[None]
        desc=torch.cat((basis/coord[None],extent/length),-1).to(p.dtype)
        weights=(mass/mass.sum()).to(p.dtype)
        scale_features=None
        scale_ids=None
        if c.multiscale_enabled:
            scale_grids=[]
            for grid_size in c.multiscale_grids:
                if grid_size==c.grid_size and geometry.get("pcraft_grid") is not None:
                    grid_np=np.asarray(geometry["pcraft_grid"],dtype=np.float32)
                else:
                    grid_np=geometry.grid(grid_size)
                if grid_np.shape!=(4,grid_size,grid_size,grid_size):
                    raise ValueError(f"invalid conservative P-CRAFT grid at resolution {grid_size}")
                scale_grids.append(torch.as_tensor(np.array(grid_np,dtype=np.float32,copy=True),
                                                   device=device).unsqueeze(0))
            scale_globals=[]; scale_maps=[]
            for grid_tensor in scale_grids:
                encode_scale=partial(self.pino_encoder.forward_with_map,preserve_resolution=True)
                if c.checkpoint_intervals and self.training and torch.is_grad_enabled():
                    pooled,fmap=checkpoint_stage("pcraft_scale_recompute",encode_scale,grid_tensor)
                else:
                    pooled,fmap=encode_scale(grid_tensor)
                scale_globals.append(self.pino_global(pooled))
                scale_maps.append(fmap)
            global_stack=torch.stack(scale_globals,dim=1)
            scale_features=global_stack.squeeze(0)
            scale_ids=torch.as_tensor(c.multiscale_grids,device=device,dtype=torch.long)
            stats_expanded=stats[:,None,:].expand(-1,len(scale_globals),-1)
            global_scores=self.pino_scale_score(torch.cat((global_stack,stats_expanded),dim=-1))
            global_weights=torch.softmax(global_scores.float(),dim=1).to(global_stack.dtype)
            pino_context=(global_stack*global_weights).sum(dim=1)
            pino_map=tuple(scale_maps)
            self.counters["multiscale_calls"]+=1
            self.counters["multiscale_global_weights"]=[float(v) for v in global_weights[0,:,0].detach().cpu()]
        else:
            grid=geometry.get("pcraft_grid")
            if grid is None: grid=geometry.grid(c.grid_size)
            grid=torch.as_tensor(np.array(grid,dtype=np.float32,copy=True),device=device).unsqueeze(0)
            if grid.shape!=(1,4,c.grid_size,c.grid_size,c.grid_size): raise ValueError("invalid P-CRAFT conservative grid")
            pino_global,pino_map=self.pino_encoder.forward_with_map(grid)
            pino_context=self.pino_global(pino_global)
            global_weights=torch.ones((1,1,1),device=device,dtype=pino_context.dtype)
            scale_features=pino_context.squeeze(0).unsqueeze(0)
            scale_ids=torch.as_tensor([c.grid_size],device=device,dtype=torch.long)
        context=(self.statistics_context(stats)+pino_context).squeeze(0)
        origin=torch.as_tensor(np.asarray(geometry["origin"],dtype=np.float64).copy(),device=device)
        span=side*float(geometry["h"])
        normalized=2*(xyz-origin)/span-1
        coords=torch.stack((normalized[:,2],normalized[:,1],normalized[:,0]),-1)
        maps=pino_map if isinstance(pino_map,(tuple,list)) else (pino_map,)
        sample_tiles=[sample_native_intervals(fmap,coords).split(chunk) for fmap in maps]
        geom_hash=str(geometry.get("geometry_sha256",""))
        if not geom_hash:
            d=sha256()
            for a in (xyz_np,mass_np,basis_np,extent_np):
                d.update(np.asarray(a,dtype="<f8").tobytes())
            if geometry.get("columns") is not None:
                d.update(np.asarray(geometry["columns"],dtype="<i8").tobytes())
            geom_hash=d.hexdigest()
        latent=torch.zeros(w,dtype=p.dtype,device=device)
        if c.gno_enabled or c.transolver_enabled or retain_interval_features:
            # Encode every native interval once. The feature table is shared
            # by local GNO messages and global physical-state slicing.
            feature_chunks=[]
            for start in range(0,len(desc),chunk):
                stop=min(start+chunk,len(desc))
                samples=tuple(parts[start//chunk] for parts in sample_tiles)
                args=(desc[start:stop],samples,context)
                if c.checkpoint_intervals and self.training and torch.is_grad_enabled():
                    h=checkpoint_stage("pcraft_interval_recompute",self._features_from_samples,*args)
                else:
                    h=self._features_from_samples(*args)
                feature_chunks.append(h)
            all_features=torch.cat(feature_chunks,dim=0)
            del feature_chunks
            if c.gno_enabled:
                neighbor_graph=self._get_neighbor_graph(geom_hash,xyz_np,device,geometry)
                extent_normalized=extent/length
                feature_tiles=all_features.split(chunk)
                updated_chunks=[]
                for i,tile in enumerate(neighbor_tiles(neighbor_graph,chunk,device)):
                    start=i*chunk
                    stop=min(start+chunk,len(desc))
                    indices=torch.as_tensor(neighbor_graph[start:stop],device=device)
                    def update_tile(center,*sources,indices=indices,tile=tile,start=start,stop=stop):
                        profile_count(gno_chunks=1)
                        idx=indices.long()
                        relative_xyz=normalized[idx]-normalized[start:stop,None,:]
                        relative_extent=torch.log((extent_normalized[idx]/extent_normalized[start:stop,None,:]).clamp_min(1e-6))
                        relative_mass=torch.log((mass[idx]/mass[start:stop,None]).clamp_min(1e-12)).unsqueeze(-1)
                        edges=torch.cat((relative_xyz,relative_extent,relative_mass),dim=-1).to(p.dtype)
                        source_table=sources[0] if len(sources)==1 else torch.cat(sources)
                        neighbors=source_table[tile.indices.long()]
                        return self._gno_update_chunk(center,neighbors,edges,weights[idx])
                    args=(feature_tiles[i],*(feature_tiles[j] for j in tile.sources))
                    if c.checkpoint_intervals and self.training and torch.is_grad_enabled():
                        updated=checkpoint_stage("pcraft_gno_recompute",update_tile,*args)
                    else:
                        updated=update_tile(*args)
                    updated_chunks.append(updated)
                all_features=torch.cat(updated_chunks)
                del feature_tiles, updated_chunks
            if c.transolver_enabled:
                with profile_stage("transolver",native_intervals=len(weights)):
                    all_features,tm=self.transolver(all_features,weights,
                    adaptive=c.transolver_adaptive,min_slices=c.transolver_min_slices,
                    adaptive_temperature=c.transolver_adaptive_temperature,
                    temperature_min=c.transolver_temperature_min,
                    temperature_max=c.transolver_temperature_max)
                self.counters["transolver_calls"]+=1
                for key,value in tm.items():
                    self.counters[key]=value
            latent=torch.matmul(weights.to(all_features.dtype),all_features)
        else:
            for start in range(0,len(desc),chunk):
                stop=min(start+chunk,len(desc))
                samples=tuple(parts[start//chunk] for parts in sample_tiles)
                args=(desc[start:stop],weights[start:stop],samples,context)
                if c.checkpoint_intervals and self.training and torch.is_grad_enabled():
                    latent=latent+checkpoint(self._chunk_pool,*args,use_reentrant=False)
                else:
                    latent=latent+self._chunk_pool(*args)
        self.counters["encode_calls"]+=1; self.counters["intervals"]+=len(xyz_np)
        cache=EncodedNativeGeometry(latent[None],stats,fixed,design,absolute,scale,coord,
            torch.stack((lo/length,hi/length)),geom_hash,side,len(xyz_np),id(self),self._versions(),
            all_features if (c.gno_enabled or c.transolver_enabled or retain_interval_features) else None,
            context,weights,desc,normalized,
            global_weights.squeeze(0) if global_weights is not None else None,
            scale_features,scale_ids)
        profile_adjoint("pcraft_native_encoding",outputs=(cache.latent,cache.interval_features,
            cache.global_context,cache.global_scale_weights,cache.scale_features),
            parameters=self.geometry_parameters())
        return cache

    def export_shared_geometry_features(self,cache: EncodedNativeGeometry):
        """Export encoded P-CRAFT interval features for a compatible SQNO branch.

        Call encode(..., retain_interval_features=True) once. Returned tensors
        retain their autograd graph, enabling joint training without running a
        second P-CRAFT geometry encoder.
        """
        if cache.owner!=id(self) or cache.versions!=self._versions():
            raise ValueError("foreign or stale geometry cache")
        required=(cache.interval_features,cache.global_context,cache.interval_mass,
                  cache.descriptor,cache.normalized_xyz,cache.global_scale_weights,
                  cache.scale_features,cache.scale_ids)
        if any(value is None for value in required):
            raise ValueError("geometry cache was not encoded with retain_interval_features=True")
        if cache.interval_features.shape[0]!=cache.native_intervals:
            raise ValueError("shared interval feature count mismatch")
        return {
            "native_feature_source":NativeFeatureSource(weakref.ref(self),cache.versions,
                not cache.latent.requires_grad and any(p.requires_grad for p in self.geometry_parameters())),
            "local_features":cache.interval_features,
            "global_context":cache.global_context,
            "global_scale_weights":cache.global_scale_weights,
            "scale_features":cache.scale_features,
            "scale_ids":cache.scale_ids,
            # SQNO consumes physical interval masses; the native cache
            # stores normalized pooling weights.
            "interval_mass":cache.interval_mass * cache.design_moments[0],
            "descriptor":cache.descriptor,
            "normalized_xyz":cache.normalized_xyz,
            "geometry_id":cache.geometry_sha256,
            "n_intervals":cache.native_intervals,
        }

    def predict(self,cache: EncodedNativeGeometry,q):
        if self.encoder_only: raise RuntimeError("encoder-only P-CRAFT has no Moment head")
        if cache.owner!=id(self) or cache.versions!=self._versions(): raise ValueError("foreign or stale geometry cache")
        q=q.to(device=cache.latent.device,dtype=torch.float64) if isinstance(q,Tensor) else torch.tensor(np.asarray(q,dtype=np.float64),device=cache.latent.device)
        single=q.ndim==1
        if single: q=q[None]
        if q.ndim!=2 or q.shape[1]!=10 or not torch.isfinite(q).all(): raise ValueError("q must be finite [Q,10]")
        maxabs=q.abs().amax(-1,keepdim=True)
        if (maxabs<=0).any(): raise ValueError("zero quadric coefficient vector")
        scaled=(q/maxabs)*cache.coordinate_scales
        unit=scaled/torch.linalg.vector_norm(scaled,dim=-1,keepdim=True)
        latent=cache.latent.expand(len(q),-1); stats=cache.statistics.expand(len(q),-1)
        if self.config.convex_potential_enabled:
            outer_grad_enabled=torch.is_grad_enabled()
            need_graph=outer_grad_enabled and q.requires_grad
            with torch.enable_grad():
                # Differentiate in monomial coordinates t=D*B*c. The output
                # remains raw moments D*grad_t(Psi), with no doubled cross
                # moments. Autograd consequently returns J_c=D*H*D*B.
                q_tilde=q*cache.coordinate_scales*self.native_B
                if not q_tilde.requires_grad:
                    q_tilde=q_tilde.detach().requires_grad_(True)
                amplitude_bound=.5*(cache.absolute_moments/cache.coordinate_scales).min()
                potential,fraction,gate_logits=self.convex_potential_head(cache.latent,q_tilde,amplitude_bound)
                gradient=torch.autograd.grad(potential.sum(),q_tilde,
                    create_graph=need_graph,retain_graph=True)[0]
                result=cache.fixed_moments+.5*cache.design_moments+cache.coordinate_scales*gradient
            self.counters["convex_potential_calls"]+=1
            self.counters["convex_potential_scale_fraction"]=float(fraction.detach())
            self.counters["convex_potential_gate_logit"]=float(gate_logits.detach())
            self.counters["convex_potential_amplitude_bound"]=float(amplitude_bound.detach())
            self.counters["convex_potential_effective_amplitude"]=float((amplitude_bound*fraction).detach())
        else:
            u=unit.to(latent.dtype)
            odd=(self.query_head(torch.cat((latent,stats,u),-1))-self.query_head(torch.cat((latent,stats,-u),-1))).double()/2
            result=cache.fixed_moments+.5*cache.design_moments+.5*cache.absolute_moments*torch.tanh(odd)
        lower,upper=_quadric_bounds(cache.normalized_bounds,unit)
        result=torch.where((lower>0)[:,None],cache.fixed_moments+cache.design_moments,result)
        result=torch.where((upper<0)[:,None],cache.fixed_moments,result)
        self.counters["queries"]+=len(q)
        return result[0] if single else result

    def convex_amplitude_diagnostics(self,cache: EncodedNativeGeometry):
        """Return interpretable convex-head gate/bound diagnostics for one geometry."""
        if not self.config.convex_potential_enabled:
            return None
        if cache.owner!=id(self) or cache.versions!=self._versions():
            raise ValueError("foreign or stale geometry cache")
        logit,fraction=self.convex_potential_head.amplitude_parameters(cache.latent)
        bound=.5*(cache.absolute_moments/cache.coordinate_scales).min()
        effective=bound*fraction
        return {
            "mode":self.config.convex_amplitude_mode,
            "gate_logit":float(logit.detach()),
            "fraction":float(fraction.detach()),
            "amplitude_bound":float(bound.detach()),
            "effective_amplitude":float(effective.detach()),
        }

    def predict_uq(self,cache: EncodedNativeGeometry,q):
        """Predict separate heteroscedastic log standard deviations for Moment and Jacobian.

        Outputs are predictive scales, not calibrated confidence until
        evaluated on held-out geometry/source-family groups.
        """
        if not self.config.uq_enabled:
            raise RuntimeError("UQ head is disabled in this checkpoint")
        if cache.owner!=id(self) or cache.versions!=self._versions():
            raise ValueError("foreign or stale geometry cache")
        value=q.to(device=cache.latent.device,dtype=torch.float64) if isinstance(q,Tensor) else torch.tensor(
            np.asarray(q,dtype=np.float64),device=cache.latent.device)
        single=value.ndim==1
        if single:value=value[None]
        if value.ndim!=2 or value.shape[1]!=10 or not torch.isfinite(value).all():
            raise ValueError("q must be finite [Q,10]")
        maxabs=value.abs().amax(-1,keepdim=True)
        if (maxabs<=0).any():raise ValueError("zero quadric coefficient vector")
        scaled=(value/maxabs)*cache.coordinate_scales
        unit=scaled/torch.linalg.vector_norm(scaled,dim=-1,keepdim=True).clamp_min(1e-30)
        latent=cache.latent.expand(len(value),-1)
        stats=cache.statistics.expand(len(value),-1)
        features=torch.cat((latent,stats,unit.to(latent.dtype)),dim=-1)
        raw=self.uq_head(features.to(dtype=self.uq_head[1].weight.dtype)).to(torch.float64)
        moment_log_std=raw[:,:10].clamp(self.config.uq_min_log_std,self.config.uq_max_log_std)
        jacobian_log_std=raw[:,10:].reshape(-1,10,10).clamp(
            self.config.uq_min_log_std,self.config.uq_max_log_std)
        result={"moment_log_std":moment_log_std,"jacobian_log_std":jacobian_log_std}
        return {key:item[0] for key,item in result.items()} if single else result

    def predict_event_validity(self,cache: EncodedNativeGeometry,q):
        """Predict whether a query lies in a derivative-valid regular regime.

        This is a learned risk signal only. It never overrides teacher
        derivative_valid labels or the exact solver's event checks.
        """
        if not self.config.event_validity_head_enabled:
            raise RuntimeError("event validity head is disabled in this checkpoint")
        if cache.owner!=id(self) or cache.versions!=self._versions():
            raise ValueError("foreign or stale geometry cache")
        q=q.to(device=cache.latent.device,dtype=torch.float64) if isinstance(q,Tensor) else torch.tensor(np.asarray(q,dtype=np.float64),device=cache.latent.device)
        single=q.ndim==1
        if single:q=q[None]
        if q.ndim!=2 or q.shape[1]!=10 or not torch.isfinite(q).all():
            raise ValueError("q must be finite [Q,10]")
        maxabs=q.abs().amax(-1,keepdim=True)
        if (maxabs<=0).any():raise ValueError("zero quadric coefficient vector")
        scaled=(q/maxabs)*cache.coordinate_scales
        unit=scaled/torch.linalg.vector_norm(scaled,dim=-1,keepdim=True)
        latent=cache.latent.expand(len(q),-1)
        stats=cache.statistics.expand(len(q),-1)
        features=torch.cat((latent,stats,unit.to(latent.dtype)),dim=-1)
        logits=self.event_validity_head(features).squeeze(-1)
        return logits[0] if single else logits

    def moments_and_jacobian(self,cache,q,*,create_graph=False):
        with torch.enable_grad():
            value=q.to(device=cache.latent.device,dtype=torch.float64) if isinstance(q,Tensor) else torch.tensor(np.asarray(q,dtype=np.float64),device=cache.latent.device)
            if not value.requires_grad: value=value.detach().requires_grad_(True)
            single=value.ndim==1
            moments=self.predict(cache,value)
            rows=moments[None] if single else moments
            jac=torch.stack([torch.autograd.grad(rows[:,i].sum(),value,create_graph=create_graph,retain_graph=True)[0] for i in range(10)],dim=-2)
        self.counters["jacobians"]+=1
        return moments,jac


def load_model(path,device="cpu"):
    payload=torch.load(path,map_location="cpu",weights_only=True)
    schema,semantics=payload.get("schema"),payload.get("semantics")
    if (schema,semantics)!=(SCHEMA,SEMANTICS) and (schema,semantics) not in LEGACY_SCHEMAS:
        raise ValueError("incompatible M80 checkpoint")
    config=dict(payload["config"])
    config.setdefault("gno_enabled",False)
    config.setdefault("gno_neighbors",8)
    config.setdefault("gno_cache_mib",64)
    config.setdefault("gno_topology_filter",False)
    config.setdefault("transolver_enabled",False)
    config.setdefault("transolver_slices",32)
    config.setdefault("transolver_heads",4)
    config.setdefault("multiscale_enabled",False)
    config.setdefault("multiscale_grids",(32,64,128))
    config.setdefault("convex_potential_enabled",False)
    if config["convex_potential_enabled"] and (schema,semantics)!=(SCHEMA,SEMANTICS):
        raise ValueError("legacy convex checkpoint has incompatible cross-coefficient semantics")
    config.setdefault("convex_potential_components",4)
    config.setdefault("convex_amplitude_mode","sigmoid")
    config.setdefault("event_validity_head_enabled",False)
    config.setdefault("uq_enabled",False)
    config.setdefault("uq_hidden",64)
    config.setdefault("uq_min_log_std",-7.0)
    config.setdefault("uq_max_log_std",4.0)
    if isinstance(config.get("multiscale_grids"),list):
        config["multiscale_grids"]=tuple(config["multiscale_grids"])
    model=PcraftNativeMomentOperator(NativePCRAFTConfig(**config))
    model.load_state_dict(payload["model"],strict=True)
    return model.to(device).eval(),payload
