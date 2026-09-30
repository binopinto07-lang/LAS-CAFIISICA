from __future__ import annotations

import numpy as np

from las_classifier.terrain.ground_debug import (
    GroundRejectReason,
    diagnose_ground_gates,
)
from las_classifier.terrain.ground_evidence import (
    GroundEvidenceConfig,
    GroundEvidenceScorer,
)


def test_ground_gate_diagnostics_and_reason_codes():
    config = GroundEvidenceConfig(
        surface_scale=0.20,
        detrend_scale=0.50,
        vertical_spread_limit=0.70,
        roughness_scale=0.20,
    )
    scorer = GroundEvidenceScorer(config)
    count = 6
    evidence = scorer.evaluate(
        ptd_score=np.array([0.90, 0.50, 0.50, 0.50, 0.50, 0.50]),
        tin_residual=np.array([0.01, 0.01, 1.00, 0.01, 0.01, 0.01]),
        vertical_residual=np.array([0.01, 0.01, 1.00, 0.01, 0.01, 0.01]),
        detrended_residual=np.zeros(count),
        neighbour_support=np.array([0.80, 0.00, 0.80, 0.01, 0.80, 0.80]),
        vertical_spread=np.zeros(count),
        roughness=np.array([0.01, 0.01, 0.01, 0.01, 1.00, 0.01]),
        original_class=np.ones(count, dtype=np.uint8),
        normal_alignment=np.array([0.90, 0.90, 0.90, 0.90, 0.90, 0.20]),
        invalid_mask=np.array([False, False, False, False, False, True]),
    )

    diagnostics, reason = diagnose_ground_gates(
        evidence,
        config,
        spatial_presence=np.array([True, False, True, True, True, True]),
    )

    assert diagnostics.no_spatial_evidence == 1
    assert diagnostics.surface_gate_fail >= 1
    assert diagnostics.spatial_gate_fail >= 1
    assert diagnostics.object_roughness >= 1
    assert diagnostics.normal_mismatch >= 1
    assert diagnostics.invalid == 1
    assert reason[0] == GroundRejectReason.ACCEPTED
    assert reason[1] == GroundRejectReason.NO_SPATIAL_EVIDENCE
    assert reason[5] == GroundRejectReason.INVALID
