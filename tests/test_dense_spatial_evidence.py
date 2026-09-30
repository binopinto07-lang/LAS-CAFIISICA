from __future__ import annotations

import numpy as np

from las_classifier.terrain.dense_spatial_evidence import DenseSpatialEvidenceGrid
from las_classifier.terrain.ground_evidence import (
    GroundEvidenceConfig,
    GroundEvidenceScorer,
)


def _grid(order: np.ndarray) -> DenseSpatialEvidenceGrid:
    x = np.array([0.10, 0.12, 0.42, 0.44, 0.78, 0.81, 1.12, 1.18, 1.45, 1.48])[order]
    y = np.array([0.10, 0.12, 0.10, 0.12, 0.10, 0.12, 0.10, 0.12, 0.10, 0.12])[order]
    z = np.array([10.0, 10.02, 10.1, 10.08, 10.2, 10.22, 10.3, 10.31, 10.4, 10.41])[order]
    ptd = np.array([0.90, 0.80, 0.91, 0.20, 0.95, 0.70, 0.88, 0.10, 0.92, 0.80])[order]
    vertical = np.array([0.01, 0.02, 0.03, 0.04, 0.02, 0.08, 0.03, 0.09, 0.01, 0.02])[order]
    plane = np.array([0.01, 0.02, 0.04, 0.10, 0.02, 0.05, 0.03, 0.12, 0.01, 0.02])[order]
    classes = np.array([2, 2, 2, 1, 2, 1, 1, 1, 2, 1], dtype=np.uint8)[order]
    rn = np.array([1, 1, 2, 1, 1, 1, 2, 1, 1, 1], dtype=np.uint8)[order]
    nr = np.array([1, 1, 2, 2, 1, 1, 2, 1, 1, 1], dtype=np.uint8)[order]
    intensity = np.arange(100, 200, 10, dtype=np.uint16)[order]

    grid = DenseSpatialEvidenceGrid.create(
        np.array([0.0, 0.0]),
        np.array([2.0, 1.0]),
        spacing=0.05,
        strong_threshold=0.68,
        max_cells=10_000,
        neighbourhood_radius=1,
    )
    for part in (slice(0, 4), slice(4, None)):
        grid.accumulate(
            x=x[part],
            y=y[part],
            z=z[part],
            ptd_score=ptd[part],
            vertical_residual=vertical[part],
            plane_distance=plane[part],
            valid_mask=np.ones(x[part].shape, dtype=np.bool_),
            original_class=classes[part],
            return_number=rn[part],
            number_of_returns=nr[part],
            intensity=intensity[part],
        )
    grid.finalize()
    return grid


def test_dense_grid_is_point_order_invariant():
    first = _grid(np.arange(10))
    second = _grid(np.array([9, 1, 6, 2, 8, 0, 5, 4, 7, 3]))

    exact = (
        "point_count",
        "strong_count",
        "min_z",
        "max_z",
        "min_vertical_residual",
        "max_vertical_residual",
        "class2_count",
        "return_only_count",
        "return_last_multi_count",
        "return_first_multi_count",
        "return_intermediate_count",
        "return_invalid_count",
        "intensity_count",
        "invalid_count",
        "intensity_histogram",
    )
    for name in exact:
        np.testing.assert_array_equal(getattr(first, name), getattr(second, name))

    np.testing.assert_allclose(first.plane_distance_sum, second.plane_distance_sum, atol=1e-12)
    np.testing.assert_allclose(first.intensity_sum, second.intensity_sum, atol=1e-12)

    x = np.array([0.11, 0.43, 0.79, 1.13, 1.47])
    y = np.full(x.shape, 0.11)
    for a, b in zip(
        first.query_with_presence(x, y),
        second.query_with_presence(x, y),
    ):
        np.testing.assert_allclose(a, b, atol=1e-7)


def test_neighbourhood_does_not_equate_missing_exact_cell_with_no_support():
    grid = DenseSpatialEvidenceGrid.create(
        np.array([0.0, 0.0]),
        np.array([2.0, 2.0]),
        spacing=0.05,
        strong_threshold=0.68,
        max_cells=10_000,
        neighbourhood_radius=1,
    )
    grid.accumulate(
        x=np.array([0.41, 0.45]),
        y=np.array([0.10, 0.12]),
        z=np.array([1.0, 1.02]),
        ptd_score=np.array([0.90, 0.90]),
        vertical_residual=np.array([0.01, 0.02]),
        plane_distance=np.array([0.01, 0.02]),
        valid_mask=np.array([True, True]),
    )

    support, spread, roughness, present = grid.query_with_presence(
        np.array([0.39]),
        np.array([0.11]),
    )
    assert present[0]
    assert support[0] == 1.0
    assert spread[0] > 0.0
    assert roughness[0] > 0.0


def test_low_z_is_neighbourhood_median_of_cell_minima():
    grid = DenseSpatialEvidenceGrid.create(
        np.array([0.0, 0.0]),
        np.array([2.0, 2.0]),
        spacing=0.05,
        strong_threshold=0.68,
        max_cells=10_000,
        neighbourhood_radius=1,
    )
    grid.accumulate(
        x=np.array([0.10, 0.45, 0.85]),
        y=np.array([0.10, 0.10, 0.10]),
        z=np.array([9.0, 10.0, 11.0]),
        ptd_score=np.array([0.9, 0.9, 0.9]),
        vertical_residual=np.zeros(3),
        plane_distance=np.zeros(3),
        valid_mask=np.ones(3, dtype=np.bool_),
    )
    low_z, present = grid.query_low_z(np.array([0.45]), np.array([0.10]))
    assert present[0]
    assert low_z[0] == 10.0


def test_grid_memory_guard_grows_cell_size_for_huge_extent():
    grid = DenseSpatialEvidenceGrid.create(
        np.array([0.0, 0.0]),
        np.array([10_000.0, 10_000.0]),
        spacing=0.05,
        strong_threshold=0.68,
        max_cells=100_000,
    )
    assert grid.cell_count <= 101_000
    assert grid.cell_size > 0.4


def test_ground_decisions_are_invariant_after_point_permutation():
    first = _grid(np.arange(10))
    second = _grid(np.array([9, 1, 6, 2, 8, 0, 5, 4, 7, 3]))

    x = np.array([0.10, 0.12, 0.42, 0.44, 0.78, 0.81, 1.12, 1.18, 1.45, 1.48])
    y = np.array([0.10, 0.12, 0.10, 0.12, 0.10, 0.12, 0.10, 0.12, 0.10, 0.12])
    ptd = np.array([0.90, 0.80, 0.91, 0.20, 0.95, 0.70, 0.88, 0.10, 0.92, 0.80])
    vertical = np.array([0.01, 0.02, 0.03, 0.04, 0.02, 0.08, 0.03, 0.09, 0.01, 0.02])
    plane = np.array([0.01, 0.02, 0.04, 0.10, 0.02, 0.05, 0.03, 0.12, 0.01, 0.02])
    classes = np.array([2, 2, 2, 1, 2, 1, 1, 1, 2, 1], dtype=np.uint8)

    config = GroundEvidenceConfig(
        surface_scale=0.28,
        detrend_scale=0.50,
        vertical_spread_limit=0.75,
        roughness_scale=0.20,
    )
    scorer = GroundEvidenceScorer(config)
    decisions = []
    ground_counts = []
    for grid in (first, second):
        neighbour, spread, roughness, presence = grid.query_with_presence(x, y)
        evidence = scorer.evaluate(
            ptd_score=ptd,
            tin_residual=plane,
            vertical_residual=vertical,
            detrended_residual=vertical,
            neighbour_support=neighbour,
            vertical_spread=spread,
            roughness=roughness,
            original_class=classes,
            spatial_presence=presence,
        )
        decisions.append(evidence.decision)
        ground_counts.append(int(np.count_nonzero(evidence.classifications() == 2)))

    np.testing.assert_array_equal(decisions[0], decisions[1])
    assert ground_counts[0] == ground_counts[1]
