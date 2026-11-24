from fractions import Fraction
import numpy as np
import pytest

from .verify import assess_event
from .bounds import Interval as I, dot, orient
from .incidence import DesignTriangles, certify_incidence


def test_observing_even_all_chart_columns_never_signs_a_neighbourhood():
    result = assess_event(scope='R8', center_signature='base', probe_signatures=['base'] * 32,
                          finite_difference_passed=True)
    assert result['event_valid'] == -1
    assert result['sample_status'] == 'all_observed_equal' and result['sampled_signature_equal'] == 1
    changed = assess_event(scope='R8', center_signature='a', probe_signatures=['b', None])
    assert changed['event_valid'] == -1
    assert changed['sample_status'] == 'observed_change' and changed['sampled_signature_equal'] == 0
    assert assess_event(scope='full')['sample_status'] == 'unmeasured'


def test_interval_tiny_exact_determinant_and_zero():
    a = np.array([1., 1., 1.])
    b = np.array([2., 2. + 2**-30, 2.])
    c = np.array([2., 2., 2. + 2**-30])
    d = np.array([3., 3., 3.])
    val = orient(*map(I.point, (a,b,c,d)))
    exact = Fraction(2,2**60)
    assert Fraction(float(val.lo)) <= exact <= Fraction(float(val.hi))
    indeterminate = dot(I.point([1.,1.,0.]),I.point([1.,-1.,0.]))
    assert indeterminate.lo <= 0 <= indeterminate.hi


def test_finite_triangle_pair_neighbourhood_and_contact():
    fixed = np.array([[0., 0., 0.], [1., 0., 0.], [0., 1., 0.]])
    moving = np.array([[.1,.2,-.2], [.9,.15,.3], [.15,.9,.2]])
    facets = np.array([[0,1,2]])
    d = DesignTriangles(fixed,facets)
    cert = certify_incidence(d,I.ball(moving, 1.e-9),facets)
    assert cert['certified'] and cert['narrow_phase_face_pairs']==1
    moving[0] = [.1,.2,0.]
    assert not certify_incidence(d,I.ball(moving,1.e-9),facets)['certified']


def test_crossing_mesh_edges_fail_closed():
    fixed = np.array([[-1.,0.,0.],[1.,0.,0.],[0.,2.,1.]])
    moving = np.array([[0.,-1.,0.],[0.,1.,0.],[2.,0.,2.]])
    faces = np.array([[0,1,2]])
    proof = certify_incidence(DesignTriangles(fixed,faces),I.ball(moving,1.e-12),faces)
    assert not proof['certified']


def test_disjoint_coplanar_faces_have_a_neighbourhood():
    fixed = np.array([[0.,0.,0.],[1.,0.,0.],[0.,1.,0.]])
    moving = fixed + np.array([1.5,1.5,0.])
    faces = np.array([[0,1,2]])
    proof = certify_incidence(DesignTriangles(fixed,faces),I.ball(moving,1.e-7),faces)
    assert proof['certified'] and proof['aabb_candidates']==0
    with pytest.raises(ValueError):
        I([0.], [np.nan])


def test_plane_enclosure_covers_perturbations_and_preserves_finite_guards():
    from .charts import enclose_plane, finite_guards
    q = np.array([.13,.4,.8,.31,0.,0.,0.,0.,0.,0.])
    radius = 2.
    center,frame = enclose_plane(q,0.,radius)
    nearby,_ = enclose_plane(q,1.e-10,radius)
    assert np.all(nearby.lo<=center.lo) and np.all(nearby.hi>=center.hi)
    tri = np.array([[0,2,1],[0,3,2],[4,5,6],[4,6,7],[0,1,5],[0,5,4],
                    [1,2,6],[1,6,5],[2,3,7],[2,7,6],[3,0,4],[3,4,7]])
    design = np.array([[-.5,-.4,-.3],[.5,.4,.3]])
    proof,_ = finite_guards(nearby,frame,radius,0,design,tri,2)
    assert proof['far_cap_clearance_lower']>0 and proof['lateral_cap_clearance_lower']>0
    with pytest.raises(ValueError):
        enclose_plane(np.r_[0.,.9,np.sqrt(1-.9**2),0.,np.zeros(6)],1.e-9,radius)


def test_regular_graph_encloses_analytic_vertices_after_pinned_eigen_residual_check():
    from .charts import enclose_checked_paraboloid
    q = np.array([.17,0.,0.,.85,.23,.51,0.,0.,0.,0.])
    raw = q/np.linalg.norm(q)
    witness = dict(eigen_verification_bound=2.**-42, least_eigenvalue_index=0,
                   canonical_quadratic_eigenvalues=[0.,raw[4],raw[5]],axis=[0.,0.,1.],
                   first=[1.,0.,0.],second=[0.,1.,0.],paraboloid_grid_resolution=24)
    pts,frame = enclose_checked_paraboloid(q,1.e-12,2.,witness)
    s,t=frame['s'],frame['t']
    expected=np.column_stack([s,t,-(q[0]+q[4]*s*s+q[5]*t*t)/q[3]])
    assert np.all(pts.lo<=expected) and np.all(pts.hi>=expected)
    with pytest.raises(ValueError):
        enclose_checked_paraboloid(q,1.e-12,2.,dict(witness,eigen_verification_bound=1.e-3))


def test_qualification_join_refuses_changed_identity_even_with_matching_query_id():
    from .verify import match_center_qualification
    import hashlib
    q=np.arange(10,dtype='<f8')
    basis=np.eye(10,dtype='<f8')[:,:3].copy()
    def digest(x):
        return hashlib.sha256(x.tobytes()).hexdigest()
    cert=dict(row_id='g_R3_004',geometry_id='g',query_id='id',scope='R3',
              raw_query_sha256=digest(q),raw_basis_sha256=digest(basis),evaluation_map_identity='map',
              source_map='v',parent_bank_sha256='bank',target_vertices_sha256='source',triangulation_hash='tri',
              operation='difference',restricted_event_valid=1,certified_raw_linf_radius=1.e-8,
              trials=[dict(raw_box_linf_radius=1.e-8,closure={'slope_lower':1.},embedding={'embedding_contraction_upper':.01},
                           mixed_incidence={'certified':True})],proof_scope='geometric incidence')
    center=dict(row_id='g_R3_004',geometry_id='g',source_record=dict(query_id='id',sha256='bank'),
                dimension=3,query=q,raw_basis=basis,context='map',primal={'vertices_sha256':'source','operation':'difference'},
                native_value={'source_map_version':'v','triangulation_hash':'tri'})
    assert match_center_qualification(cert,center)['restricted_event_valid']==1
    for mutated in (dict(center,context='other'),dict(center,query=-q),dict(center,raw_basis=basis[:,::-1])):
        with pytest.raises(ValueError):
            match_center_qualification(cert,mutated)
    assert match_center_qualification(None,center)['restricted_event_valid']==-1


def test_growth_sweep_rejects_wrong_larger_cap_at_same_center():
    from .charts import enclose_plane, finite_guards
    q=np.array([.13,.4,.8,.31,0.,0.,0.,0.,0.,0.])
    pts,frame=enclose_plane(q,1.e-12,2.)
    faces=np.array([[0,2,1],[0,3,2],[4,5,6],[4,6,7],[0,1,5],[0,5,4],
                    [1,2,6],[1,6,5],[2,3,7],[2,7,6],[3,0,4],[3,4,7]])
    with pytest.raises(ValueError,match='earlier cap'):
        finite_guards(pts,frame,2.,1,np.array([[-.5,-.5,-.5],[.5,.5,.5]]),faces,2)


def test_norm_lower_endpoint_for_zero_residual_interval():
    from .bounds import norm
    residual = I([-1.e-15]*3,[1.e-15]*3)
    assert norm(residual).lo==0 and norm(residual).hi > np.sqrt(3.)*1.e-15


def test_exhaustive_r8_admission_requires_an_exact_registered_geometry_and_full_cover():
    from copy import deepcopy
    import json
    from zipfile import ZipFile
    from .admission import PROOF_DIR, check_cover
    name = 'R02_R8_032.json'
    actual = json.loads((PROOF_DIR / name).read_bytes())
    with ZipFile(PROOF_DIR / 'replay_inputs.zip') as archive:
        raw = json.loads(archive.read('geometry/' + name))
    assert check_cover(actual, raw) == 2.**-52
    bad = deepcopy(actual)
    bad['quadratic_product_shape_per_normalizer'][0][0] -= 1
    with pytest.raises(ValueError, match='classifier cover'):
        check_cover(bad, raw)
    with pytest.raises(ValueError, match='different finite operand'):
        check_cover(dict(actual, cap_mask_sha256='different'), raw)
    bad = deepcopy(actual)
    bad['raw_axes_hex'][0][0] = bad['raw_axes_hex'][0][1]
    with pytest.raises(ValueError, match='raw axis'):
        check_cover(bad, raw)
    with pytest.raises(ValueError, match='original geometry proof'):
        check_cover(actual, dict(raw, material_mesh_faces_sha256='different'))


def test_conditional_r8_geometry_with_center_jacobian_stays_unknown():
    import json
    from zipfile import ZipFile
    from .admission import PROOF_DIR
    from .verify import match_center_qualification
    name = 'R02_R8_032.json'
    with ZipFile(PROOF_DIR / 'replay_inputs.zip') as archive:
        raw = json.loads(archive.read('geometry/' + name))
        center = json.loads(archive.read('centers/' + name))
    assert raw['checked_source_envelope_incidence'] and center['material']['restricted_valid']
    assert match_center_qualification(raw, center)['restricted_event_valid'] == -1
    with pytest.raises(ValueError, match='missing linked exhaustive'):
        match_center_qualification(dict(raw, restricted_event_valid=1,
                                        certified_raw_linf_radius=2.**-52), center)
