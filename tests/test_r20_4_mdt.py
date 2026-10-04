"""R20.4 MDT provenance tests: interpolation is never measured Ground."""
from __future__ import annotations

import numpy as np

from las_classifier.terrain.mdt_export import (
    INTERPOLATED_MDT,
    NO_GROUND_OBSERVATION,
    OBSERVED_GROUND,
    fill_small_mdt_gaps,
)


def test_small_gap_is_marked_interpolated_not_observed():
    z = np.full((5, 5), np.nan, dtype=np.float32)
    observed = np.zeros((5, 5), dtype=bool)
    z[2, 1] = 10.0
    z[2, 3] = 12.0
    observed[2, 1] = True
    observed[2, 3] = True

    filled, state = fill_small_mdt_gaps(
        z, observed, resolution_m=0.25, max_gap_m=0.30,
    )
    assert state[2, 1] == OBSERVED_GROUND
    assert state[2, 2] == INTERPOLATED_MDT
    assert np.isfinite(filled[2, 2])


def test_large_unobserved_area_stays_no_ground_observation():
    z = np.full((9, 9), np.nan, dtype=np.float32)
    observed = np.zeros((9, 9), dtype=bool)
    z[0, 0] = 100.0
    observed[0, 0] = True

    _, state = fill_small_mdt_gaps(
        z, observed, resolution_m=0.25, max_gap_m=0.50,
    )
    assert state[8, 8] == NO_GROUND_OBSERVATION
