from __future__ import annotations

import numpy as np

from las_classifier.terrain.ground_evidence import (
    GroundDecision,
    GroundEvidenceConfig,
    GroundEvidenceScorer,
)


def _evaluate(original_class):
    scorer = GroundEvidenceScorer(
        GroundEvidenceConfig(
            surface_scale=0.28,
            detrend_scale=0.50,
            vertical_spread_limit=0.75,
            roughness_scale=0.20,
        )
    )
    count = 4
    return scorer.evaluate(
        ptd_score=np.array(
            [0.90, 0.86, 0.30, 0.78]
        ),
        tin_residual=np.array(
            [0.04, 0.06, 1.20, 0.08]
        ),
        vertical_residual=np.array(
            [0.03, 0.04, 1.40, 0.05]
        ),
        detrended_residual=np.array(
            [0.05, 0.08, 1.10, 0.10]
        ),
        neighbour_support=np.array(
            [0.90, 0.80, 0.05, 0.72]
        ),
        vertical_spread=np.array(
            [0.10, 0.12, 2.00, 0.15]
        ),
        roughness=np.array(
            [0.03, 0.04, 0.60, 0.05]
        ),
        original_class=np.asarray(
            original_class,
            dtype=np.uint8,
        ),
        return_number=np.array(
            [2, 2, 2, 1],
            dtype=np.uint8,
        ),
        number_of_returns=np.array(
            [2, 2, 2, 2],
            dtype=np.uint8,
        ),
        intensity_score=np.full(
            count,
            0.60,
        ),
    )


def test_class2_is_low_prior_not_truth():
    all_class2 = _evaluate(
        [2, 2, 2, 2]
    )
    no_class2 = _evaluate(
        [1, 1, 1, 1]
    )

    delta = np.abs(
        all_class2.score
        - no_class2.score
    )
    assert np.all(
        delta <= 0.03001
    )
    assert (
        all_class2.decision[2]
        != GroundDecision.L3_GROUND_ORIGINAL_VALIDATED
    )
    assert (
        no_class2.decision[0]
        == GroundDecision.L3_GROUND_RECOVERED_HIGH
    )


def test_last_return_is_evidence_not_automatic_ground():
    evidence = _evaluate(
        [1, 1, 1, 1]
    )

    assert (
        evidence.return_evidence[2]
        == 1.0
    )
    assert evidence.decision[2] in {
        GroundDecision.NON_GROUND_VEGETATION,
        GroundDecision.NON_GROUND_OBJECT,
    }
    assert (
        evidence.classifications()[2]
        == 1
    )


def test_recovered_ground_keeps_measured_provenance():
    evidence = _evaluate(
        [2, 1, 1, 1]
    )
    source = (
        evidence.ground_source_codes()
    )

    assert source[0] == 1
    assert source[1] in {3, 4}
    assert source[2] == 0
