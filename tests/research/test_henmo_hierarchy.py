"""Material, finite-operand and derivative controls for the HENMO hierarchy."""
import numpy as np
import pytest

from spin_adapter.henmo_cubature import compress_positive_measure, moment_basis
from spin_adapter.henmo_hierarchy import (
    SparseMaterialHierarchy, CoarseCubatureUnavailable, evaluate_deterministic, compare_deterministic_rules,
)
from spin_adapter.v21_material import prepare_rle_material
from spin_adapter.v21_blocks import BlockMeasure
from spin_adapter.v21_response import prepare_surface


def box(lo, hi):
    v = np.asarray([[lo[0],lo[1],lo[2]],[hi[0],lo[1],lo[2]],
        [hi[0],hi[1],lo[2]],[lo[0],hi[1],lo[2]],
        [lo[0],lo[1],hi[2]],[hi[0],lo[1],hi[2]],
        [hi[0],hi[1],hi[2]],[lo[0],hi[1],hi[2]]],dtype=float)
    f = np.array([[0,2,1],[0,3,2],[4,5,6],[4,6,7],[0,1,5],[0,5,4],
                  [1,2,6],[1,6,5],[2,3,7],[2,7,6],[3,0,4],[3,4,7]])
    return v,f


def cube_material(side=8):
    runs = np.array([[x,y,0,side] for x in range(side) for y in range(side)],dtype=np.int32)
    return prepare_rle_material(runs,[0,0,0],1/side,side)


def cut_at(cut=.5):
    v,f=box([-2,-2,-2],[cut,2,2])
    velocity=np.zeros((8,3,3))
    velocity[v[:,0]==cut,0,0]=1
    return prepare_surface(v,f),velocity


def test_sparse_tree_aggregates_parent_children_and_covers_each_leaf_once():
    material=cube_material()
    tree=SparseMaterialHierarchy.from_material(material,leaf_grid=4)
    np.testing.assert_allclose(tree.total_moments,material.moments(),atol=2e-15)
    assert len(tree.nodes)==1+8+64
    for node in tree.nodes:
        if node.children:
            np.testing.assert_allclose(node.moments,
                sum((tree.nodes[i].moments for i in node.children),np.zeros(10)),atol=2e-15)
    surface,_=cut_at(.37)
    selected=tree.select(surface,.03,stop_grid=2)
    leaves=np.concatenate([tree.leaf_indices(i) for i in
        selected.retained_nodes+selected.discarded_nodes+selected.boundary_nodes])
    np.testing.assert_array_equal(np.sort(leaves),np.arange(64))
    assert all(tree.nodes[i].grid<=2 for i in selected.boundary_nodes)


def test_full_empty_root_certificates_include_finite_end_caps():
    tree=SparseMaterialHierarchy.from_material(cube_material(),leaf_grid=4)
    full=tree.select(prepare_surface(*box([-2]*3,[3]*3)),.1)
    assert full.retained_nodes==(0,) and full.distance_queries==1
    # The cap above this bounded operand excludes the entire material. An
    # infinite x halfspace would give a different answer.
    empty=tree.select(prepare_surface(*box([-2]*3,[3,3,-1])),.1)
    assert empty.discarded_nodes==(0,) and not empty.boundary_nodes
    flipped=tree.select(prepare_surface(*box([-2]*3,[3,3,-1])),.1,'difference')
    np.testing.assert_allclose(flipped.retained_moments,tree.total_moments)


def test_hierarchy_matches_existing_flat_execution_with_same_atoms_and_motion(tmp_path):
    from spin_adapter.compact_pino_v21 import evaluate_packed
    blocks=BlockMeasure.build(cube_material(),tmp_path/'pack',block_voxels=2)
    tree=SparseMaterialHierarchy.from_blocks(blocks)
    surface,motion=cut_at(.37)
    fixed=np.arange(10)*.003
    for operation in ('intersection','difference'):
        old=evaluate_packed(blocks,surface,.13,operation,fixed,motion,full_cell_rule='gl3')
        new=evaluate_deterministic(tree,surface,.13,operation,fixed,motion,full_cell_rule='gl3')
        np.testing.assert_allclose(new.moments.detach(),old.moments.detach(),rtol=3e-14,atol=2e-14)
        np.testing.assert_allclose(new.jacobian.detach(),old.jacobian.detach(),rtol=3e-13,atol=2e-14)


def test_same_finite_surface_complement_adds_fixed_material_only_once_per_query():
    tree=SparseMaterialHierarchy.from_material(cube_material(),leaf_grid=4)
    surface,motion=cut_at(.37)
    fixed=np.arange(10)*.007
    a=evaluate_deterministic(tree,surface,.13,'intersection',fixed,motion)
    b=evaluate_deterministic(tree,surface,.13,'difference',fixed,motion)
    np.testing.assert_allclose(a.moments+b.moments,2*fixed+tree.total_moments,atol=2e-14)
    np.testing.assert_allclose(a.jacobian+b.jacobian,0,atol=2e-14)


def test_no_false_derivative_plateau_at_artificial_block_interface():
    tree=SparseMaterialHierarchy.from_material(cube_material(),leaf_grid=2)
    surface,motion=cut_at(.5)
    result=evaluate_deterministic(tree,surface,.125,vertex_velocity=motion,full_cell_rule='gl3')
    # GL3 integrates the cubic transition times degree-two basis exactly here.
    assert result.moments[0].item()==pytest.approx(.5,abs=1e-14)
    assert result.jacobian[0,0].item()==pytest.approx(1.,abs=1e-13)
    h=1e-6
    plus=evaluate_deterministic(tree,cut_at(.5+h)[0],.125,full_cell_rule='gl3')
    minus=evaluate_deterministic(tree,cut_at(.5-h)[0],.125,full_cell_rule='gl3')
    np.testing.assert_allclose((plus.moments-minus.moments)/(2*h),result.jacobian[:,0],
                               rtol=3e-8,atol=3e-10)


@pytest.mark.parametrize('side',[512,1024])
def test_high_native_resolution_build_uses_runs_not_expanded_cells(side):
    material=prepare_rle_material(np.array([[side//2,side//2,0,side]],np.int32),
        [0,0,0],1/side,side)
    def forbidden(*args,**kwargs):
        raise AssertionError('full material atoms must not be expanded during hierarchy build')
    material.iter_chunks=forbidden
    tree=SparseMaterialHierarchy.from_material(material,leaf_grid=64)
    assert len(tree.leaf_nodes)==64 and len(tree.nodes)<512
    assert tree.diagnostics['native_side']==side
    assert tree.diagnostics['dense_native_grid'] is False
    np.testing.assert_allclose(tree.total_moments,material.moments(),rtol=2e-14,atol=1e-20)
    compressed=tree.measure(0,max_atoms=32)
    assert compressed.source_count==4
    assert compressed.source_rule=='analytic_full_runs_tetra4_and_original_partial_atoms'
    np.testing.assert_allclose(moment_basis(compressed.nodes).T@compressed.weights,
                               tree.total_moments,rtol=2e-14,atol=1e-20)


def test_positive_compression_preserves_full_moments_and_uses_actual_source_positions():
    rng=np.random.default_rng(17)
    nodes=rng.uniform(-1,1,(400,3))
    weights=rng.uniform(.1,2,400)
    rule=compress_positive_measure(nodes,weights,max_atoms=40)
    assert rule.compressed and len(rule.weights)<=40 and np.all(rule.weights>0)
    np.testing.assert_array_equal(rule.nodes,nodes[rule.source_indices])
    np.testing.assert_allclose(moment_basis(rule.nodes).T@rule.weights,
                               moment_basis(nodes).T@weights,rtol=3e-12,atol=3e-11)
    assert rule.normalized_moment_error<2e-11
    assert not rule.nodes.flags.writeable
    with pytest.raises(ValueError,match='at least ten'):
        compress_positive_measure(nodes,weights,max_atoms=9)
    with pytest.raises(ValueError,match='positive'):
        compress_positive_measure(nodes,-weights,max_atoms=40)


def test_coarse_run_rule_preserves_complete_moments_but_does_not_claim_cut_equivalence():
    material=cube_material(16)
    tree=SparseMaterialHierarchy.from_material(material,leaf_grid=4)
    coarse=tree.measure(0,max_atoms=40)
    assert coarse.source_count==4*16*16
    assert coarse.source_count<material.node_count
    np.testing.assert_allclose(moment_basis(coarse.nodes).T@coarse.weights,
                               tree.total_moments,rtol=3e-12,atol=3e-12)
    assert 'full_runs' in coarse.source_rule


def test_streamed_compression_stays_bounded_and_preserves_complete_moments():
    material=cube_material(32)
    tree=SparseMaterialHierarchy.from_material(material,leaf_grid=4)
    rule=tree.measure(0,max_atoms=32,chunk_size=113)
    assert rule.source_count==4*32*32 and len(rule.nodes)<=32
    np.testing.assert_allclose(moment_basis(rule.nodes).T@rule.weights,
                               tree.total_moments,rtol=3e-11,atol=3e-11)
    with pytest.raises(ValueError):
        rule.weights.flags.writeable=True
    with pytest.raises(ValueError):
        tree.nodes[0].moments.flags.writeable=True
    tree.clear_cache()
    other=tree.measure(0,max_atoms=32,chunk_size=65536)
    np.testing.assert_array_equal(rule.nodes,other.nodes)
    np.testing.assert_array_equal(rule.weights,other.weights)


def test_uncompressible_coarse_measure_reports_fallback_without_discarding_material(monkeypatch):
    import spin_adapter.henmo_hierarchy as implementation
    from spin_adapter.henmo_cubature import CompressedCubature
    material=cube_material(40)
    tree=SparseMaterialHierarchy.from_material(material,leaf_grid=4)
    largest=[]
    def unreduced(points,weights,max_atoms):
        largest.append(len(weights))
        return CompressedCubature(points,weights,np.arange(len(weights)),
            moment_basis(points).T@weights,len(weights),0.,False,'SIMULATED_NUMERICAL_FAILURE')
    monkeypatch.setattr(implementation,'compress_positive_measure',unreduced)
    with pytest.raises(CoarseCubatureUnavailable) as failure:
        tree.measure(0,max_atoms=32,chunk_size=1024)
    assert max(largest)<=4096
    assert failure.value.source_count<=4096
    assert 'SIMULATED_NUMERICAL_FAILURE' in str(failure.value)


def test_spatial_features_preserve_subregion_information_and_sum_full_moments():
    material=cube_material(8)
    tree=SparseMaterialHierarchy.from_material(material,leaf_grid=4)
    feature=tree.spatial_features(0,grid=2)
    np.testing.assert_allclose(feature[:,0],np.full(8,.125),atol=3e-15)
    np.testing.assert_allclose(feature.sum(0),[1,0,0,0,1/12,1/12,1/12,0,0,0],atol=3e-15)
    assert tree.spatial_features(0,grid=2) is feature


def test_material_atoms_and_descriptors_share_one_byte_budget():
    tree=SparseMaterialHierarchy.from_material(cube_material(8),leaf_grid=4,cache_bytes=700)
    rule=tree.measure(0,max_atoms=12)
    assert tree.diagnostics['atom_cache_bytes']>0
    feature=tree.spatial_features(0,grid=2)
    assert tree.diagnostics['cache_bytes']<=700
    assert tree.diagnostics['feature_cache_bytes']==feature.nbytes
    second=tree.measure(0,max_atoms=12)
    np.testing.assert_array_equal(rule.nodes,second.nodes)
    np.testing.assert_array_equal(rule.weights,second.weights)
    assert tree.diagnostics['cache_bytes']<=700
    assert tree.diagnostics['cache_bytes']==tree.diagnostics['atom_cache_bytes']+tree.diagnostics['feature_cache_bytes']
    tree.clear_cache()
    assert tree.diagnostics['cache_bytes']==0


def test_material_replacement_and_writable_storage_are_rejected():
    material=cube_material(4)
    tree=SparseMaterialHierarchy.from_material(material,leaf_grid=2)
    # Immutable byte storage cannot be made writable through a child view.
    with pytest.raises(ValueError):
        material.full_runs.flags.writeable=True
    material.full_runs=material.full_runs.copy()
    with pytest.raises(ValueError,match='writable|replaced'):
        tree.select(cut_at()[0],.1)


def test_multiple_static_hierarchies_can_share_one_immutable_material():
    material=cube_material(4)
    a=SparseMaterialHierarchy.from_material(material,leaf_grid=2)
    b=SparseMaterialHierarchy.from_material(material,leaf_grid=4)
    a.assert_source_unchanged(); b.assert_source_unchanged()
    np.testing.assert_allclose(a.total_moments,b.total_moments,atol=2e-15)


def test_packed_provenance_replacement_is_rejected(tmp_path):
    blocks=BlockMeasure.build(cube_material(4),tmp_path/'pack',block_voxels=2)
    tree=SparseMaterialHierarchy.from_blocks(blocks)
    blocks.material=cube_material(4)
    with pytest.raises(ValueError,match='provenance'):
        tree.measure(0,max_atoms=32)


def test_mapped_material_change_is_rejected_without_full_query_hash(tmp_path):
    blocks=BlockMeasure.build(cube_material(4),tmp_path/'pack',block_voxels=2)
    tree=SparseMaterialHierarchy.from_blocks(blocks)
    path=tmp_path/'pack'/'nodes.npy'
    import os
    stat=path.stat()
    os.utime(path,ns=(stat.st_atime_ns,stat.st_mtime_ns+1_000_000))
    with pytest.raises(ValueError,match='replaced'):
        tree.select(cut_at()[0],.1)


def test_rule_comparison_reports_empirical_error_not_false_certificate():
    tree=SparseMaterialHierarchy.from_material(cube_material(4),leaf_grid=2)
    surface,motion=cut_at(.47)
    prediction,report=compare_deterministic_rules(tree,surface,.13,vertex_velocity=motion)
    assert len(report['records'])==3
    assert report['discrepancy_is_error_certificate'] is False
    assert prediction.diagnostics['exact_hard_mesh_oracle'] is False
    assert report['records'][-1]['moment_discrepancy']>=0
    assert report['records'][-1]['jacobian_discrepancy']>=0
