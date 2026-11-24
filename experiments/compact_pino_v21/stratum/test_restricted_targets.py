import pytest
from experiments.compact_pino_v21.stratum.restricted_targets import masks


def row():
    return dict(dimension=3, material=dict(moment_valid=True,restricted_valid=True,moment_tetra={'passed':True},
                jacobian_tetra={'passed':True}),same_primal_motion_source=True,sampled_row_passed=True,
                steps=[dict(h=h,all_columns_passed=True,columns=[dict(raw_col=i,source_combinatorics_stable=True,
                       source_velocity={'passed':True},material_jacobian={'passed':True},
                       retained={'complete':True,'velocity':{'passed':True}}) for i in range(3)])
                       for h in (2e-5,1e-5)])


def test_finite_samples_never_promote_an_unknown_event():
    proof=dict(restricted_event_valid=-1,ambient_event_valid=-1,native_output_triangulation_event_valid=-1)
    quality=masks(row(),proof,(2e-5,1e-5))
    assert quality['all_material_fd'] and quality['all_sampled_checks']
    assert not quality['qualified_material_j'] and not quality['qualified_with_all_sampled_checks']


def test_numerical_and_retained_topology_are_separate_requirements():
    data=row()
    proof=dict(restricted_event_valid=1,ambient_event_valid=-1,native_output_triangulation_event_valid=-1)
    data['steps'][0]['columns'][1]['retained']['complete']=False
    with pytest.raises(ValueError,match='archived step or row'):
        masks(data,proof,(2e-5,1e-5))
    data['steps'][0]['all_columns_passed']=False
    data['sampled_row_passed']=False
    quality=masks(data,proof,(2e-5,1e-5))
    assert quality['qualified_material_j'] and not quality['all_retained_correspondences']
    assert not quality['qualified_with_all_sampled_checks'] and not quality['ambient_event_certified']
    data['steps'][1]['h']=2e-5
    assert not masks(data,proof,(2e-5,1e-5))['qualified_material_j']
