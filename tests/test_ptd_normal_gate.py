from __future__ import annotations

import numpy as np

from las_classifier.classifiers.adaptive_ptd import _axelsson_acceptance
from las_classifier.ground.terrain_tin import TerrainTIN


def test_ptd_candidate_normal_gate_rejects_surface_incompatible_point():
    vertices = np.array(
        [
            [0.0, 0.0, 0.0],
            [2.0, 0.0, 1.0],
            [0.0, 2.0, 0.0],
            [2.0, 2.0, 1.0],
        ],
        dtype=np.float64,
    )
    tin = TerrainTIN.build(vertices)

    candidates = np.array(
        [
            [0.6, 0.6, 0.30],
            [1.2, 1.2, 0.60],
        ],
        dtype=np.float64,
    )

    terrain_normal = np.array(
        [-0.5, 0.0, 1.0],
        dtype=np.float64,
    )
    terrain_normal /= np.linalg.norm(terrain_normal)

    candidate_normals = np.vstack(
        (
            terrain_normal,
            np.array([0.0, 1.0, 0.0], dtype=np.float64),
        )
    )

    accepted = _axelsson_acceptance(
        tin,
        candidates,
        max_distance=0.10,
        max_angle_deg=15.0,
        min_triangle_edge=0.10,
        candidate_normals=candidate_normals,
        min_normal_alignment=0.58,
    )

    assert bool(accepted[0])
    assert not bool(accepted[1])


def test_ptd_normal_gate_does_not_reject_when_normals_are_unknown():
    vertices = np.array(
        [
            [0.0, 0.0, 0.0],
            [2.0, 0.0, 1.0],
            [0.0, 2.0, 0.0],
            [2.0, 2.0, 1.0],
        ],
        dtype=np.float64,
    )
    tin = TerrainTIN.build(vertices)
    candidates = np.array(
        [[0.6, 0.6, 0.30]],
        dtype=np.float64,
    )
    normals = np.full((1, 3), np.nan, dtype=np.float64)

    accepted = _axelsson_acceptance(
        tin,
        candidates,
        max_distance=0.10,
        max_angle_deg=15.0,
        min_triangle_edge=0.10,
        candidate_normals=normals,
    )

    assert bool(accepted[0])
