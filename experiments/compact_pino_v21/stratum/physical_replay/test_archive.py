"""A copied or hand-edited published aggregate must agree with every underlying measured column."""
from copy import deepcopy

import pytest

from . import archive


def candidate():
    columns=[{"passed_sampled_column":True} for _ in range(3)]
    step={"n_planned":3,"n_attempted":3,"columns":columns,"all_columns_passed":True}
    return {"row_id":"manufactured", "dimension":3,"same_primal_motion_source":True,
            "material":{"moment_tetra":{"passed":True},"jacobian_tetra":{"passed":True}},
            "steps":[deepcopy(step),deepcopy(step)],"sampled_row_passed":True}


@pytest.mark.parametrize("tamper",["positive_step","negative_step","positive_row","negative_row","missing_column"])
def test_rejects_corrupt_sampled_aggregate(tamper):
    row=candidate()
    if tamper=="positive_step":
        row["steps"][0]["columns"][1]["passed_sampled_column"]=False
    elif tamper=="negative_step":
        row["steps"][0]["all_columns_passed"]=False
    elif tamper=="positive_row":
        row["material"]["jacobian_tetra"]["passed"]=False
    elif tamper=="negative_row":
        row["sampled_row_passed"]=False
    elif tamper=="missing_column":
        row["steps"][0]["columns"].pop()
    with pytest.raises(ValueError,match="aggregate"):
        archive.check_sampled_aggregates(row)


def test_preserves_negative_when_any_registered_direction_failed():
    row=candidate()
    row["steps"][0]["columns"][1]["passed_sampled_column"]=False
    row["steps"][0]["all_columns_passed"]=False
    row["sampled_row_passed"]=False
    archive.check_sampled_aggregates(row)


def test_verifies_published_triangulation_changes_against_observed_incidence():
    signature = dict(unique_constructions=True, vertex_count=3, construction_sha256="support",
                     unoriented_incidence_sha256="triangles", oriented_incidence_sha256="winding")
    row = dict(row_id="manufactured", material={"retained_signature": signature})
    flipped = dict(signature, oriented_incidence_sha256="opposite")
    column = dict(endpoints=[row, {"material": {"retained_signature": flipped}}],
                  retained=dict(n_center=3, n_matched=3, same_construction_set=True,
                                same_unoriented_incidence=True, same_oriented_incidence=False, complete=False))
    archive.check_discrete_receipts(row, column)
    column["retained"]["complete"] = True
    with pytest.raises(ValueError, match="signed boundary signatures"):
        archive.check_discrete_receipts(row, column)
