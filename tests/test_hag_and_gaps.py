from __future__ import annotations

import numpy as np

from las_classifier.ground.gap_detection import (
    OCCLUDED_GAP,
    detect_gaps,
)
from las_classifier.ground.height_above_ground import (
    hag_confidence,
    height_above_ground,
)
from las_classifier.ground.terrain_tin import TerrainTIN


def test_hag_uses_local_tin_on_steep_plane():
    axis = np.arange(0.0, 5.01, 1.0)
    x, y = np.meshgrid(axis, axis)
    z = 20.0 + 1.8 * x + 0.35 * y
    vertices = np.column_stack((x.ravel(), y.ravel(), z.ravel()))
    tin = TerrainTIN.build(vertices)

    qx = np.array([1.5, 2.5, 3.5])
    qy = np.array([1.5, 2.5, 3.5])
    terrain = 20.0 + 1.8 * qx + 0.35 * qy
    qz = terrain + np.array([0.02, 1.20, 3.00])

    hag, distance = height_above_ground(tin, qx, qy, qz)
    score = hag_confidence(
        hag,
        positive_limit=0.35,
        negative_limit=0.60,
    )

    assert abs(float(hag[0]) - 0.02) < 0.03
    assert float(hag[1]) > 1.0
    assert float(hag[2]) > 2.8
    assert float(distance[0]) < 0.03
    assert float(score[0]) > 0.95
    assert float(score[1]) < 0.01
    assert float(score[2]) < 1e-6


def test_gap_detection_marks_tree_occlusion_as_supported_gap():
    grid = np.arange(0.0, 9.0, 1.0)
    x, y = np.meshgrid(grid, grid)
    z = 100.0 + 0.20 * x + 0.10 * y

    hole = (
        (x >= 3.0)
        & (x <= 5.0)
        & (y >= 3.0)
        & (y <= 5.0)
    )
    vertices = np.column_stack((x[~hole], y[~hole], z[~hole]))
    tin = TerrainTIN.build(vertices)

    evidence = np.array(
        [
            [4.0, 4.0, 104.0],
            [4.1, 4.0, 103.5],
            [3.9, 4.1, 102.8],
        ],
        dtype=np.float64,
    )

    gaps = detect_gaps(
        tin,
        dense_edge=1.6,
        max_gap_edge=5.0,
        evidence_xyz=evidence,
        occlusion_hag=0.50,
    )

    assert gaps.detected_count > 0
    assert gaps.supported_count > 0
    assert gaps.occluded_count > 0
    assert np.any(gaps.kind == OCCLUDED_GAP)
