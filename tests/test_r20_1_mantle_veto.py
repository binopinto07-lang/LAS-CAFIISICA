"""R20.1 regression: roofs and canopy are NOT Ground even if R20 follows them.

Use synthetic measured aggregates, including a deliberately wrong PTD prior.
This checks point-order-independent, bounded-array spatial veto logic.
"""
from __future__ import annotations

import numpy as np

from las_classifier.terrain.dense_spatial_evidence import DenseSpatialEvidenceGrid
from las_classifier.terrain.inverted_mantle import build_inverted_ground_mantle
from las_classifier.terrain.mantle_export import (
    mantle_state_grid, STATE_ROOF_CANDIDATE, STATE_CANOPY_CANDIDATE,
)
from las_classifier.terrain.mantle_veto import (
    VETO_CANOPY_CANDIDATE,
    VETO_HEIGHT,
    VETO_NONE,
    VETO_ROOF_CANDIDATE,
    build_mantle_veto,
)


def _scene(*, roof=False, canopy=False, terrace=False, slope=False):
    grid = DenseSpatialEvidenceGrid.create(
        np.array([0.0, 0.0]), np.array([20.0, 20.0]),
        spacing=0.05, strong_threshold=0.68, max_cells=10_000,
    )
    ny, nx = grid.ny, grid.nx
    yy, xx = np.indices((ny, nx))
    floor = np.full((ny, nx), 10.0, dtype=np.float32)
    if slope:
        floor += 0.18 * xx * grid.cell_size + 0.12 * yy * grid.cell_size
    cell_spread = np.full((ny, nx), 0.04, dtype=np.float32)
    if roof:
        floor[20:30, 20:30] += 4.5
    if canopy:
        floor[20:28, 20:28] += 3.5
        cell_spread[20:28, 20:28] = 1.90
    if terrace:
        # An elongated terrace has lower terrain on one/two sides, not an
        # isolated elevated object surrounded in three cardinal directions.
        floor[21:27, :] += 2.5

    counts = np.full((ny, nx), 4, dtype=np.int32)
    grid.point_count[:] = counts.ravel()
    grid.geometry_count[:] = counts.ravel()
    grid.strong_count[:] = 3  # Deliberately wrong PTD prior over the roof.
    grid.min_z[:] = floor.ravel()
    grid.max_z[:] = (floor + cell_spread).ravel()
    grid.min_vertical_residual[:] = 0.0
    grid.max_vertical_residual[:] = 0.02
    grid.plane_distance_sum[:] = 0.02 * grid.point_count
    grid.ptd_score_sum[:] = 0.95 * grid.point_count
    grid.processed_points = grid.measured_point_count
    return grid


def _xy(grid, ix, iy):
    return (
        np.array([(ix + 0.5) * grid.cell_size]),
        np.array([(iy + 0.5) * grid.cell_size]),
    )


def test_roof_is_vetoed_when_cloth_follows_roof_even_with_wrong_ptd_class2():
    grid = _scene(roof=True)
    mantle = build_inverted_ground_mantle(grid)
    guard = build_mantle_veto(grid, mantle)
    x, y = _xy(grid, 25, 25)
    rooftop = 14.5
    # R20 alone may accept the measured roof as a near-mantle point;
    # R20.1 must recognise its elevation relative to surrounding terrain.
    assert guard.roof_candidate[25, 25]
    assert guard.classify_veto(x, y, np.array([rooftop]))[0] == VETO_ROOF_CANDIDATE
    ox, oy = _xy(grid, 7, 7)
    assert guard.classify_veto(ox, oy, np.array([10.0]))[0] == VETO_NONE


def test_tree_canopy_is_vetoed_when_lower_envelope_follows_foliage():
    grid = _scene(canopy=True)
    mantle = build_inverted_ground_mantle(grid)
    guard = build_mantle_veto(grid, mantle)
    x, y = _xy(grid, 24, 24)
    assert guard.canopy_candidate[24, 24]
    assert guard.classify_veto(x, y, np.array([13.5]))[0] == VETO_CANOPY_CANDIDATE


def test_above_ground_vegetation_is_vetoed_even_with_ground_below_in_cell():
    grid = _scene()
    # Same XY cell contains a genuine low return and vegetation 1.5m higher.
    ix, iy = 25, 25
    key = ix + iy * grid.nx
    grid.max_z[key] = grid.min_z[key] + 1.50
    mantle = build_inverted_ground_mantle(grid)
    guard = build_mantle_veto(grid, mantle)
    x, y = _xy(grid, ix, iy)
    assert guard.classify_veto(x, y, np.array([11.5]))[0] == VETO_HEIGHT
    assert guard.classify_veto(x, y, np.array([10.0]))[0] == VETO_NONE


def test_long_terrace_and_sloping_ground_are_not_roof_candidates():
    for grid in (_scene(terrace=True), _scene(slope=True)):
        mantle = build_inverted_ground_mantle(grid)
        guard = build_mantle_veto(grid, mantle)
        x, y = _xy(grid, 25, 24)
        if np.isclose(grid.min_z[25 + 24 * grid.nx], 12.5):
            z = 12.5
        else:
            z = float(grid.min_z[25 + 24 * grid.nx])
        assert guard.classify_veto(x, y, np.array([z]))[0] == VETO_NONE


def test_unknown_and_outside_cloud_is_not_automatically_rejected():
    grid = _scene()
    mantle = build_inverted_ground_mantle(grid)
    guard = build_mantle_veto(grid, mantle)
    codes = guard.classify_veto(
        np.array([-500.0, np.nan, 200.0]),
        np.array([10.0, 10.0, 200.0]),
        np.array([20.0, 20.0, 20.0]),
    )
    np.testing.assert_array_equal(codes, np.zeros(3, dtype=np.uint8))


def test_guard_results_do_not_depend_on_query_point_order():
    grid = _scene(roof=True)
    mantle = build_inverted_ground_mantle(grid)
    guard = build_mantle_veto(grid, mantle)
    xa, ya = _xy(grid, 25, 25)
    xb, yb = _xy(grid, 7, 7)
    x = np.r_[xa, xb]
    y = np.r_[ya, yb]
    z = np.array([14.5, 10.0])
    expected = guard.classify_veto(x, y, z)
    order = np.array([1, 0])
    actual = guard.classify_veto(x[order], y[order], z[order])
    np.testing.assert_array_equal(expected[order], actual)


def test_r20_1_mantle_view_warns_roof_and_canopy_without_classifying_them_ground():
    for options, centre, expected in (
        ({"roof": True}, (25, 25), STATE_ROOF_CANDIDATE),
        ({"canopy": True}, (24, 24), STATE_CANOPY_CANDIDATE),
    ):
        grid = _scene(**options)
        mantle = build_inverted_ground_mantle(grid)
        mantle.veto_guard = build_mantle_veto(grid, mantle)
        states = mantle_state_grid(mantle)
        assert states[centre[1], centre[0]] == expected
        assert not np.any(states == 2)  # Dense synthetic scene: no invented holes.
