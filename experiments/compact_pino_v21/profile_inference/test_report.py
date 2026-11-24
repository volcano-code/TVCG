"""The seed-zero rejection belongs to wall-clock accounting in both passes."""
import pytest

from experiments.compact_pino_v21.profile_inference.report import summarize, pair_archive


def test_all_requests_contribute_to_timing_and_rejections_are_paired():
    rows = [dict(call=1, query=[1], defined=True, callback_wall_s=2., source_wall_s=1.,
                 surface_index_wall_s=.2, solve_response_wall_s=.8),
            dict(call=2, query=[2], defined=False, error='event', callback_wall_s=8., source_wall_s=8.)]
    archive = [dict(index=1, q=[1], seconds=3., mass=1.), dict(index=2, q=[2], seconds=.25, error='event')]
    assert summarize(rows)['all_callback_wall_sum_s'] == 10.
    assert summarize(rows)['accepted_response_median_s'] == .8
    assert summarize(rows)['rejected_calls'] == [2]
    pair_archive(rows, archive)
    assert rows[1]['archived_native_joint_s'] == .25
    rows[1]['query'] = [3]
    with pytest.raises(ValueError):
        pair_archive(rows, archive)
