"""Independent manufactured checks for rank paths, construction keys and numeric acceptance."""

import numpy as np

from . import checks

def test_retraction_preserves_rotating_null_axis_and_raw_column():
    q = np.array([.2, .3, .4, .8, 2., 1., 0., 0., 0., 0.])
    b = np.array([-.3, .4, -.5, .6, .3, -.1, 0., -.8, .7, .2])
    p = checks.retracted(q, b, 1.e-5, "PARABOLOID")
    m = checks.retracted(q, b, -1.e-5, "PARABOLOID")
    assert abs(np.linalg.det(checks.block(p))) < 1e-14
    np.testing.assert_allclose((p-m)/2e-5, b, rtol=0, atol=2e-9)
    assert checks.rank_signature(q, "PARABOLOID") == checks.rank_signature(p, "PARABOLOID")

def test_reordered_facet_soup_preserves_construction_topology_but_flip_changes_it():
    faces = np.array([[0,2,1],[0,1,3],[1,2,3],[2,0,3]])
    keys = np.array([[0,21],[0,32],[0,43],[0,54]], dtype=np.int64)
    order = np.array([2,3,0,1])
    pa = checks.retain_keys(keys, faces)
    pb = checks.retain_keys(keys[order], np.argsort(order)[faces[::-1, [1, 2, 0]]])
    assert pa.topology == pb.topology
    assert pa.constructions == pb.constructions
    assert pa.winding == pb.winding
    assert np.array_equal(order[pb.order], pa.order)
    reverse = checks.retain_keys(keys, faces[:, ::-1])
    assert reverse.winding != pa.winding
    assert reverse.topology == pa.topology

def test_different_construction_and_missing_vertices_cannot_certify_velocity():
    keys = np.array([[0,21],[0,32],[0,43],[0,54]], dtype=np.int64)
    f = np.array([[0,2,1],[0,1,3],[1,2,3],[2,0,3]])
    a = checks.retain_keys(keys, f)
    b = checks.retain_keys(np.array([[0,21],[0,32],[0,43],[0,55]]), f)
    x = np.array([[1.,3.,4.],[3.,.1,2.],[2.,4.,6.],[-1.,4.,1.]])
    delta = np.arange(12).reshape(4,3)/12.
    h=1e-5
    audit=checks.compare_retained(a,b,a,x-delta*h,x+delta*h,delta,h)
    assert not audit["complete"]
    assert audit["n_matched"] == 3
    assert audit["velocity"]["max_abs"] < 5e-10

def test_signed_tetrahedra_match_translated_scaled_tetrahedron():
    from .checks import tetra_pair
    vertices=np.array([[0.,0.,0.],[1.,0.,0.],[0.,1.,0.],[0.,0.,1.]])
    faces=np.array([[0,2,1],[0,1,3],[1,2,3],[2,0,3]])
    motion=np.stack([vertices,np.ones_like(vertices)*[1,0,0],np.zeros_like(vertices)],axis=2)
    fixed=np.array([.1,0,0,0,0,0,0,0,0,0])
    m,j=tetra_pair(vertices,faces,motion,2.,fixed)
    editable=np.array([1/3,1/12,1/12,1/12,1/30,1/30,1/30,1/60,1/60,1/60])
    np.testing.assert_allclose(m, editable+fixed, atol=1e-14)
    np.testing.assert_allclose(j[:,0], editable*np.array([3,4,4,4,5,5,5,5,5,5]),atol=1e-14)
    np.testing.assert_allclose(j[:,1],[0,1/3,0,0,1/6,0,0,0,1/12,1/12],atol=1e-14)
    assert not checks.error_metric(j[:,1]+.01, j[:,1], "material")["passed"]
    assert checks.error_metric(np.zeros(10), np.zeros(10), "material")["passed"]


def test_retraction_uses_supplied_binary_center_and_holds_its_small_eigenvalue():
    q = np.array([.25,.33,-.81,.19,1.5,-.56,3e-14,0.,0.,0.])
    direction = np.array([0.,.3,.4,-.1,.1,-.2,0.,.23,-.14,.9])
    np.testing.assert_array_equal(checks.retracted(q,direction,0.,"PARABOLOID").view("u8"),q.view("u8"))
    varied=checks.retracted(q,direction,1e-5,"PARABOLOID")
    eig=np.linalg.eigvalsh(checks.block(varied))
    assert abs(eig[np.argmin(abs(eig))]-3e-14)<4e-16


def test_cherchi_support_keeps_the_lpi_line_and_plane_roles_distinct():
    def code(keys):
        return checks._identity(dict(point_type="LPI",operand_coordinates_exact=keys),int)
    assert code([2,1,5,4,3])==code([1,2,3,4,5])
    assert code([3,2,1,4,5])!=code([1,2,3,4,5])


def test_nonunique_design_source_vertex_does_not_gain_a_correspondence():
    import pytest
    index=checks.PhysicalVertexIDs.index_design([[1.,2.,3.],[0.,3.,4.]])
    ids=checks.PhysicalVertexIDs(index, [[0.,3.,4.],[9.,0.,1.]])
    assert ids.at((1.,2.,3.))==0
    assert ids.at((9.,0.,1.))==3
    with pytest.raises(ValueError,match="ambiguous"):
        ids.at([0.,3.,4.])
