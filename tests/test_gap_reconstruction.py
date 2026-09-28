from __future__ import annotations

import numpy as np

from las_classifier.ground.gap_reconstruction import (
    estimate_fill_count,
    iter_triangle_fill,
)
from las_classifier.ground.terrain_tin import TerrainTIN


def _long_thin_tin() -> TerrainTIN:
    vertices = np.array(
        [
            [0.0, 0.0, 0.0],
            [20.0, 0.0, 0.0],
            [20.0, 0.10, 0.0],
            [0.0, 0.10, 0.0],
        ],
        dtype=np.float64,
    )
    return TerrainTIN.build(vertices)


def test_generated_fill_count_matches_estimate():
    tin = _long_thin_tin()
    mask = np.ones(tin.triangle_count, dtype=np.bool_)

    expected = estimate_fill_count(tin, mask, 0.50)
    chunks = list(iter_triangle_fill(tin, mask, 0.50))
    actual = sum(int(x.size) for x, _, _ in chunks)

    assert expected > 0
    assert actual == expected


def test_fill_count_respects_limit():
    tin = _long_thin_tin()
    mask = np.ones(tin.triangle_count, dtype=np.bool_)

    expected = estimate_fill_count(
        tin,
        mask,
        0.02,
        max_points=17,
    )
    chunks = list(
        iter_triangle_fill(
            tin,
            mask,
            0.02,
            max_points=17,
            chunk_points=5,
        )
    )
    actual = sum(int(x.size) for x, _, _ in chunks)

    assert expected == 17
    assert actual == 17
