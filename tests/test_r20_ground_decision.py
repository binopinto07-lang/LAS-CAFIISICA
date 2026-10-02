"""R20 decision codes must remain separate from inferred mantle cells."""
from __future__ import annotations

import numpy as np

from las_classifier.terrain.ground_debug import GroundRejectReason, diagnose_ground_gates
from las_classifier.terrain.ground_evidence import (
    GroundDecision,
    GroundEvidenceConfig,
    GroundEvidenceScorer,
)


def test_mantle_measured_return_has_distinct_source_and_is_accepted():
    config = GroundEvidenceConfig(
        surface_scale=0.20,
        detrend_scale=0.50,
        vertical_spread_limit=0.75,
        roughness_scale=0.20,
    )
    evidence = GroundEvidenceScorer(config).evaluate(
        ptd_score=np.array([0.0, 0.98]),
        tin_residual=np.array([np.inf, 0.0]),
        vertical_residual=np.array([np.nan, 0.0]),
        detrended_residual=np.array([np.nan, 0.0]),
        neighbour_support=np.array([0.0, 0.98]),
        vertical_spread=np.zeros(2),
        roughness=np.zeros(2),
        original_class=np.array([1, 2], dtype=np.uint8),
    )
    evidence.decision[0] = int(GroundDecision.L3_GROUND_MANTLE_RECOVERED)
    assert evidence.classifications()[0] == 2
    assert evidence.ground_source_codes()[0] == 5
    assert evidence.ground_source_codes()[1] != 5
    diag, reasons = diagnose_ground_gates(
        evidence, config, spatial_presence=np.ones(2, dtype=np.bool_)
    )
    assert reasons[0] == GroundRejectReason.ACCEPTED
    assert diag.rejected_total == 0
