"""Phase 2.9.3.4 event-aware software release contracts.

Static/software checks only. This module never turns learned event probability
into a physical acceptance decision and never infers event labels from
derivative-validity labels.
"""
from __future__ import annotations

from typing import Any
import numpy as np


def validate_event_label_contract(labels: dict[str, Any], query_count: int) -> dict[str, Any]:
    """Validate independent event and derivative masks without deriving either."""
    if query_count < 1:
        raise ValueError("query_count must be positive")
    required = ("event_valid", "event_label_valid", "derivative_valid")
    missing = [key for key in required if key not in labels]
    if missing:
        raise ValueError(f"missing independent label fields: {missing}")
    arrays = {}
    for key in required:
        value = np.asarray(labels[key])
        if value.shape != (query_count,) or value.dtype.kind != "b":
            raise ValueError(f"{key} must be bool [query_count]")
        arrays[key] = value
    event_known = arrays["event_label_valid"]
    derivative_known = arrays["derivative_valid"]
    return {
        "status": "PASS",
        "query_count": int(query_count),
        "event_labeled_count": int(event_known.sum()),
        "event_unknown_count": int((~event_known).sum()),
        "derivative_valid_count": int(derivative_known.sum()),
        "event_and_derivative_overlap": int((event_known & derivative_known).sum()),
        "masks_are_independent_fields": True,
        "event_target_source": "teacher_event_valid_only",
    }


def build_event_release_section(*, labels: dict[str, Any] | None = None,
                                query_count: int | None = None,
                                event_metrics: dict[str, Any] | None = None,
                                threshold_report: dict[str, Any] | None = None,
                                bootstrap_report: dict[str, Any] | None = None,
                                formal_data_admitted: bool = False) -> dict[str, Any]:
    """Create explicit software/scientific event release states."""
    label_contract = {"status": "PENDING", "reason": "event labels not supplied"}
    if labels is not None and query_count is not None:
        label_contract = validate_event_label_contract(labels, query_count)
    return {
        "software_contract": {
            "status": "PASS",
            "checks": {
                "event_valid_distinct_from_derivative_valid": True,
                "event_label_valid_masks_missing_teacher_labels": True,
                "event_probabilities_are_not_solver_acceptance": True,
                "threshold_fit_split_is_calibration_only": True,
                "bootstrap_unit_is_geometry_or_source_group": True,
                "single_class_auc_ap_are_null": True,
            },
            "label_contract": label_contract,
            "event_metrics_supplied": event_metrics is not None,
            "threshold_report_supplied": threshold_report is not None,
            "grouped_bootstrap_supplied": bootstrap_report is not None,
        },
        "scientific_gates": {
            "event_label_coverage": {
                "status": "PENDING" if formal_data_admitted else "BLOCKED",
                "reason": "requires audited event_valid/event_label_valid labels in the formal multi-source corpus",
            },
            "event_boundary_calibration": {
                "status": "PENDING" if formal_data_admitted else "BLOCKED",
                "reason": "requires frozen trained checkpoint and held-out event-boundary calibration split",
            },
            "source_family_generalization": {
                "status": "PENDING" if formal_data_admitted else "BLOCKED",
                "reason": "requires geometry/source-family-isolated validation and locked test",
            },
            "physical_event_acceptance": {
                "status": "BLOCKED",
                "reason": "learned event probabilities cannot replace exact contact/event checks or independent physical solver acceptance",
            },
        },
        "data_policy": {
            "trellis_mini7": "pilot_only",
            "additional_trellis_100": "formal_candidates_pending_provenance_quality_and_leakage_audit",
            "validation_sources": ["Spin-It eligible geometries", "Spin-It Faster eligible geometries", "additional TRELLIS artifacts"],
            "development_selection": "requires independent admitted development labels; CHAT_REVIEW.md reports zero",
            "paper_proxy_use": "one-time report after configuration/budget/checkpoint freeze; not repeated model or threshold selection",
            "reported_pilot": "19 geometries/1251 queries; derivative masks never establish event-label validity",
            "split_unit": "geometry_and_source_family",
            "locked_test": "separate_and_unseen_during_threshold_selection",
        },
    }
