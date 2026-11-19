import numpy as np
import pytest

from research.NeuralMoments.models.pcraft_ufno.release_gate.event_release import (
    validate_event_label_contract, build_event_release_section,
)


def test_event_release_requires_three_distinct_teacher_masks():
    labels = {
        "event_valid": np.array([True, False, True], dtype=bool),
        "event_label_valid": np.array([True, True, False], dtype=bool),
        "derivative_valid": np.array([False, True, True], dtype=bool),
    }
    report = validate_event_label_contract(labels, 3)
    assert report["status"] == "PASS"
    assert report["event_labeled_count"] == 2
    assert report["derivative_valid_count"] == 2
    assert report["event_and_derivative_overlap"] == 1
    assert report["masks_are_independent_fields"] is True


def test_event_release_rejects_missing_or_malformed_masks():
    with pytest.raises(ValueError, match="missing independent"):
        validate_event_label_contract({"event_valid": np.array([True])}, 1)
    with pytest.raises(ValueError, match="must be bool"):
        validate_event_label_contract({
            "event_valid": np.array([1]), "event_label_valid": np.array([True]),
            "derivative_valid": np.array([False]),
        }, 1)


def test_event_release_keeps_formal_scientific_gates_blocked_without_data():
    section = build_event_release_section()
    assert section["software_contract"]["status"] == "PASS"
    assert section["scientific_gates"]["event_label_coverage"]["status"] == "BLOCKED"
    assert section["scientific_gates"]["event_boundary_calibration"]["status"] == "BLOCKED"
    assert section["scientific_gates"]["source_family_generalization"]["status"] == "BLOCKED"
    assert section["scientific_gates"]["physical_event_acceptance"]["status"] == "BLOCKED"


def test_event_release_preserves_dataset_policy_and_locked_test_separation():
    section = build_event_release_section()
    policy = section["data_policy"]
    assert policy["trellis_mini7"] == "pilot_only"
    assert "pending" in policy["additional_trellis_100"]
    assert len(policy["validation_sources"]) == 3
    assert policy["split_unit"] == "geometry_and_source_family"
    assert "unseen" in policy["locked_test"]


def test_event_release_can_report_label_coverage_without_claiming_calibration():
    labels = {
        "event_valid": np.array([True, False], dtype=bool),
        "event_label_valid": np.array([True, False], dtype=bool),
        "derivative_valid": np.array([True, True], dtype=bool),
    }
    section = build_event_release_section(labels=labels, query_count=2)
    assert section["software_contract"]["label_contract"]["event_labeled_count"] == 1
    assert section["scientific_gates"]["event_boundary_calibration"]["status"] == "BLOCKED"
