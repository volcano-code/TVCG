import json
import numpy as np
import pytest
from .corpus import cell_moments, generate_corpus


def test_independent_cell_integral_and_clipped_box():
    # Box [0,2]x[0,3]x[0,4], density2, volume24.
    lower=np.array([[0.,0.,0.]]);upper=np.array([[2.,3.,4.]])
    expected=np.array([48,48,72,96,64,144,256,144,96,72.])
    np.testing.assert_allclose(cell_moments(lower,upper,np.array([2.])),expected,rtol=1e-14)
    # Half x box [1,2] => known mean x1.5 and E[x²]=7/3.
    expected=np.array([24,36,36,48,56,72,128,72,72,54.])
    lower[:,0]=1
    np.testing.assert_allclose(cell_moments(lower,upper,np.array([2.])),expected,rtol=1e-14)


def test_corpus_split_and_roundtrip_without_m83(tmp_path):
    target=tmp_path/'density-v1'
    manifest=generate_corpus(target)
    assert len(manifest['records'])==12
    assert [x['split'] for x in manifest['records']].count('train')==8
    assert [x['split'] for x in manifest['records']].count('validation')==2
    assert [x['split'] for x in manifest['records']].count('held_out')==2
    for rec in manifest['records']:
        with np.load(target/rec['file'],allow_pickle=False) as z:
            assert z['density'].std()>0
            assert np.all(z['density']<=z['base_density'])
            assert not np.any(z['fixed'] & z['editable'])
            np.testing.assert_allclose(z['moments'][1],z['fixed_moments'])
            assert np.all(z['moments'][:,0]>=z['fixed_moments'][0]-1e-12)
            assert np.all(z['moments'][:,0]<=z['moments'][0,0]+1e-12)
    assert json.loads((target/'manifest.json').read_text())==manifest
    with pytest.raises(FileExistsError): generate_corpus(target)
