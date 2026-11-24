import numpy as np
import pytest

from experiments.compact_pino_v21.stratum.prototype import compare_derived


def test_portable_rescoring_allows_float_roundoff_and_preserves_screen_decisions():
    saved = {'p95': 0.051496, 'n': 25, 'above': ['combB_R8_036', 'combB_R8_037']}
    compare_derived(saved, dict(saved, p95=float(np.nextafter(saved['p95'], 0))), 'R8')
    for changed in (dict(saved, p95=0.05), dict(saved, n=24), dict(saved, above=['combB_R8_036'])):
        with pytest.raises(ValueError):
            compare_derived(saved, changed, 'R8')
