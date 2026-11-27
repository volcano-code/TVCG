"""One-time conservative inputs and reusable P-CRAFT features for HENMO.

The native material side is independent of both the context and feature grids.
PreparedMaterial uses analytic run aggregation, so a 1024 material does not
require a 1024-cubed dense tensor or a complete expanded point pack.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import numpy as np
import torch

from .henmo_hierarchy import SparseMaterialHierarchy
from .henmo_cubature import moment_basis
from .v21_material import PreparedMaterial


def geometry_versions(model):
    result = []
    for prefix, module in (('encoder',model.shared_encoder),('summary',model.geometry_attention),
                           ('local_geometry',getattr(model,'local_geometry',None))):
        if module is None:
            continue
        for kind, values in (('parameter',module.named_parameters()),('buffer',module.named_buffers())):
            for name,p in values:
                result.append((prefix+'.'+kind+'.'+name,id(p),p._version,p.data_ptr(),p.dtype,p.device,p.requires_grad))
    return tuple(result)


class GraphLifetime:
    def __init__(self):
        self.consumed = False

    def backward(self, value):
        self.consumed = True
        return value


@dataclass(frozen=True)
class MaterialHandle:
    identity: str
    side: int
    pitch: float
    origin: np.ndarray


@dataclass(frozen=True)
class HENMOGeometry:
    hierarchy: SparseMaterialHierarchy
    blocks: object
    fixed_moments: np.ndarray
    table: torch.Tensor
    feature_grid: int
    owner: int
    versions: tuple
    table_version: int
    fixed_bytes: bytes
    inference_only: bool
    graph_lifetime: GraphLifetime | None
    node_contexts: dict = field(default_factory=dict, compare=False)
    node_context_bytes: dict = field(default_factory=lambda: {'bytes':0}, compare=False)

    @property
    def nbytes(self):
        return (self.table.numel()*self.table.element_size()+self.fixed_moments.nbytes
                +self.node_context_bytes['bytes'])

    def context(self, points):
        tree = self.hierarchy
        origin = torch.tensor(tree.origin.copy(), dtype=points.dtype, device=points.device)
        index = ((points-origin) * (self.feature_grid/(tree.side*tree.pitch))).floor().long()
        index = index.clamp(0, self.feature_grid-1)
        keys = (index[:,0]*self.feature_grid + index[:,1])*self.feature_grid + index[:,2]
        return self.table[keys]


def make_hierarchy(source, config):
    if isinstance(source, SparseMaterialHierarchy):
        tree = source
    elif isinstance(source, PreparedMaterial):
        tree = SparseMaterialHierarchy.from_material(source, leaf_grid=config.leaf_grid,
                                                     cache_bytes=config.material_cache_bytes)
    elif hasattr(source, 'block_voxels'):
        if source.side // source.block_voxels == config.leaf_grid:
            tree = SparseMaterialHierarchy.from_blocks(source, cache_bytes=config.material_cache_bytes)
        elif source.material is not None:
            tree = SparseMaterialHierarchy.from_material(source.material, leaf_grid=config.leaf_grid,
                                                         cache_bytes=config.material_cache_bytes)
        else:
            raise ValueError('changing packed leaf resolution requires original material provenance')
    else:
        raise ValueError('expected original PreparedMaterial, BlockMeasure or material hierarchy')
    if tree.side != config.native_material_resolution or tree.leaf_grid != config.leaf_grid:
        raise ValueError('actual material/leaf resolution differs from HENMO scientific config')
    tree.assert_source_unchanged()
    return tree


def aggregate_grid(tree, grid):
    """Positive, physical ten-moment rows on a chosen *context* grid."""
    if type(grid) is not int or grid < 1 or tree.side % grid:
        raise ValueError('aggregation grid must divide native material side')
    table = np.zeros((grid**3, 10), dtype=np.float64)
    if tree.material is not None:
        indices, moments = tree.material.block_moments(tree.side//grid)
        table[np.ravel_multi_index(indices.T, (grid,)*3)] = moments
    elif grid <= tree.leaf_grid:
        centers = np.array([tree.nodes[i].center for i in tree.leaf_nodes])
        index = np.floor((centers-tree.origin) * grid/(tree.pitch*tree.side)).astype(np.int64)
        keys = np.ravel_multi_index(index.T, (grid,)*3)
        np.add.at(table, keys, np.array([tree.nodes[i].moments for i in tree.leaf_nodes]))
    else:
        # Legacy packs may lack runs. Scan their existing positive atoms only
        # during preparation, never at each query and never invent fine support.
        for points, weights in tree.chunks(tree.root, max_nodes=131072, full_cell_rule='tetra4'):
            index = np.floor((points-tree.origin)*grid/(tree.pitch*tree.side)).astype(np.int64).clip(0,grid-1)
            keys = np.ravel_multi_index(index.T, (grid,)*3)
            values = moment_basis(points)*weights[:,None]
            for k in range(10):
                table[:,k] += np.bincount(keys, weights=values[:,k], minlength=grid**3)
    return table


def encoder_grid(tree, grid, density):
    from scipy.ndimage import binary_erosion, distance_transform_edt
    width = tree.side*tree.pitch/grid
    fraction = aggregate_grid(tree, grid)[:,0].reshape((grid,)*3)/(density*width**3)
    if not np.isfinite(fraction).all() or fraction.min() < -1e-12 or fraction.max() > 1+1e-8:
        raise ValueError('material density or material volume fraction is inconsistent')
    fraction = np.clip(fraction,0.,1.)
    occupied = fraction > 0
    signed = (-np.ones_like(fraction) if occupied.all() else
              distance_transform_edt(~occupied)-distance_transform_edt(occupied))
    boundary = occupied & ~binary_erosion(occupied, structure=np.ones((3,)*3), border_value=0)
    return np.stack((fraction, occupied, np.tanh(signed/2), boundary)).astype(np.float32)


def encode_table(model, tree, fixed, density):
    config = model.config
    grid = config.feature_grid
    masses = aggregate_grid(tree, grid)
    active = np.flatnonzero(masses[:,0] > 0)
    m = masses[active]
    editable = m.sum(0)
    span = tree.side*tree.pitch
    coord = span**np.array([0,1,1,1,2,2,2,2,2,2])
    means = m/m[:,:1]
    width = span/grid
    centers = tree.origin+(np.array(np.unravel_index(active,(grid,)*3)).T+.5)*width
    low, high = centers.min(0)-width/2, centers.max(0)+width/2
    scale = (editable[0]+fixed[0])*coord
    if scale[0] <= 0:
        raise ValueError('positive total material mass required')
    features = np.column_stack((means/coord,np.full((len(m),3),width/span)))
    stats = np.r_[(editable+fixed)/scale, fixed/scale, editable/scale,
                  (low+high)/(2*span),(high-low)/span]
    coords = (2*(means[:,1:4]-tree.origin)/span-1)[:,[2,1,0]].copy()
    encoder = model.shared_encoder
    parameter = next(encoder.parameters())
    def tensor(value):
        return torch.tensor(value, dtype=parameter.dtype, device=parameter.device)
    maps, globals_ = [], []
    for context_grid in config.context_grids:
        image = encoder_grid(tree, context_grid, density)
        pooled, feature_map = encoder.pino_encoder.forward_with_map(tensor(image)[None], preserve_resolution=True)
        maps.append(feature_map)
        globals_.append(encoder.pino_global(pooled))
    glob, s = torch.stack(globals_,dim=1), tensor(stats)[None]
    score = encoder.pino_scale_score(torch.cat((glob,s[:,None].expand(-1,len(maps),-1)),-1))
    shared = encoder.statistics_context(s)+(glob*score.softmax(1)).sum(1)
    local = encoder._interval_features(tensor(features),tensor(coords),maps,shared)
    log_mass = tensor(np.log(m[:,0]/editable[0]))[None]
    summary,_ = model.geometry_attention(shared[:,None],local[None],local[None],
                                         attn_mask=log_mass,need_weights=False)
    return torch.index_copy(local.new_zeros((grid**3,local.shape[1])),0,
                            torch.tensor(active,device=parameter.device),local+summary[0])
