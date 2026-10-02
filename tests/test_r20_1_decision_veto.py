"""R20.1 integration: rejected roofs cannot survive a high PTD/class2 score."""
from __future__ import annotations

import numpy as np

from las_classifier.terrain.ground_debug import GroundRejectReason, diagnose_ground_gates
from las_classifier.terrain.ground_evidence import (
    GroundDecision,
    GroundEvidenceConfig,
    GroundEvidenceScorer,
    PROV_MANTLE_VETO,
)
from las_classifier.terrain.mantle_veto import (
    VETO_NONE, VETO_HEIGHT, VETO_ROOF_CANDIDATE, VETO_CANOPY_CANDIDATE,
    apply_mantle_veto,
)


class FixedGuard:
    def classify_veto(self, x, y, z):
        return np.array(
            [VETO_ROOF_CANDIDATE, VETO_CANOPY_CANDIDATE, VETO_HEIGHT, VETO_ROOF_CANDIDATE, VETO_NONE],
            dtype=np.uint8,
        )


def test_roof_and_canopy_override_high_ptd_source_class2_and_do_not_reaccept():
    cfg = GroundEvidenceConfig(
        surface_scale=0.20,
        detrend_scale=0.50,
        vertical_spread_limit=0.75,
        roughness_scale=0.20,
    )
    n = 5
    evidence = GroundEvidenceScorer(cfg).evaluate(
        ptd_score=np.full(n, 0.98),
        tin_residual=np.zeros(n),
        vertical_residual=np.zeros(n),
        detrended_residual=np.zeros(n),
        neighbour_support=np.ones(n),
        vertical_spread=np.zeros(n),
        roughness=np.zeros(n),
        original_class=np.full(n, 2, dtype=np.uint8),
    )
    assert np.all(evidence.classifications() == 2)
    actual = apply_mantle_veto(
        evidence,
        FixedGuard(),
        np.zeros(n), np.zeros(n), np.zeros(n),
        np.array([False, False, False, True, False]),
    )
    np.testing.assert_array_equal(
        actual, np.array([2, 3, 1, 0, 0], dtype=np.uint8)
    )
    assert evidence.decision[0] == GroundDecision.NON_GROUND_OBJECT
    assert evidence.decision[1] == GroundDecision.NON_GROUND_VEGETATION
    assert evidence.decision[2] == GroundDecision.NON_GROUND_OBJECT
    assert evidence.decision[3] == GroundDecision.L3_GROUND_ORIGINAL_VALIDATED
    assert evidence.decision[4] == GroundDecision.L3_GROUND_ORIGINAL_VALIDATED
    assert np.count_nonzero(evidence.classifications() == 2) == 2
    assert np.all((evidence.provenance[:3] & PROV_MANTLE_VETO) != 0)

    diagnostics, reasons = diagnose_ground_gates(evidence, cfg)
    assert diagnostics.rejected_total == 3
    assert reasons[0] == GroundRejectReason.ROOF_CANDIDATE_VETO
    assert reasons[1] == GroundRejectReason.CANOPY_CANDIDATE_VETO
    assert reasons[2] == GroundRejectReason.MANTLE_HEIGHT_VETO
    assert reasons[3] == GroundRejectReason.ACCEPTED
