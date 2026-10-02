"""R20.2 regression tests: measured gaps, vineyard faces and object veto."""
from __future__ import annotations

import numpy as np

from las_classifier.terrain.dense_spatial_evidence import DenseSpatialEvidenceGrid
from las_classifier.terrain.ground_continuity import (
    apply_continuity_recovery, build_ground_continuity,
)
from las_classifier.terrain.ground_debug import GroundRejectReason, diagnose_ground_gates
from las_classifier.terrain.ground_evidence import (
    GroundDecision, GroundEvidenceConfig, GroundEvidenceScorer,
    PROV_GROUND_CONTINUITY,
)
from las_classifier.terrain.inverted_mantle import (
    InvertedGroundMantle, MantleConfig,
)
from las_classifier.terrain.mantle_veto import (
    MantleVeto, MantleVetoConfig, VETO_HEIGHT, VETO_NONE,
    VETO_ROOF_CANDIDATE,
)


def _synthetic_scene(*, slope: float = 0.0):
    grid = DenseSpatialEvidenceGrid.create(
        np.array([0.0, 0.0]), np.array([11.0, 11.0]),
        spacing=0.05, strong_threshold=0.68, max_cells=10_000,
    )
    ny, nx = grid.ny, grid.nx
    yy, xx = np.indices((ny, nx))
    lower = 10.0 + slope * (xx + 0.5) * grid.cell_size
    grid.point_count[:] = 4
    grid.geometry_count[:] = 4
    grid.strong_count[:] = 3
    grid.min_z[:] = lower.ravel()
    grid.max_z[:] = (lower + 0.04).ravel()
    observed = np.ones((ny, nx), dtype=np.bool_)
    reliable = np.ones((ny, nx), dtype=np.bool_)
    # A measured strip unclassified by R20's PTD/TIN and therefore previously
    # black. It is present physically, but carries no trusted PTD geometry.
    reliable[12, 3:-3] = False
    grid.geometry_count.reshape(ny, nx)[12, 3:-3] = 0
    grid.strong_count.reshape(ny, nx)[12, 3:-3] = 0
    surface = lower.astype(np.float32)
    gx = np.full((ny, nx), slope, dtype=np.float32)
    gy = np.zeros((ny, nx), dtype=np.float32)
    zero = np.zeros((ny, nx), dtype=np.bool_)
    mantle = InvertedGroundMantle(
        origin=grid.origin.copy(), cell_size=grid.cell_size, nx=nx, ny=ny,
        surface=surface, observed=observed, reliable=reliable,
        inferred=zero.copy(), ambiguous=~reliable,
        possible_no_ground_observation=zero.copy(),
        anchor_distance=np.zeros((ny, nx), dtype=np.float32),
        slope_x=gx, slope_y=gy,
        reference_z=30.0, config=MantleConfig(),
    )
    guard = MantleVeto(
        origin=grid.origin.copy(), cell_size=grid.cell_size, nx=nx, ny=ny,
        surface=surface, reliable=reliable, slope_x=gx, slope_y=gy,
        roof_candidate=zero.copy(), canopy_candidate=zero.copy(),
        directional_lower_count=np.zeros((ny, nx), dtype=np.uint8),
        config=MantleVetoConfig(),
    )
    return grid, mantle, guard


def _xyz(grid, ix, iy, *, z=None):
    x = np.array([(ix + 0.5) * grid.cell_size], dtype=np.float64)
    y = np.array([(iy + 0.5) * grid.cell_size], dtype=np.float64)
    if z is None:
        z = float(grid.min_z[ix + iy * grid.nx])
    return x, y, np.array([z], dtype=np.float64)


def test_recover_physically_measured_strip_rejected_by_original_mantle():
    grid, mantle, guard = _synthetic_scene()
    continuity = build_ground_continuity(grid, mantle, guard)
    x, y, z = _xyz(grid, 10, 12)
    assert not mantle.recovery_mask(x, y, z)[0]  # R20: unreliable cell
    assert continuity.connected[12, 10]
    assert continuity.expanded_cell_count >= 1
    assert continuity.recovery_mask(x, y, z)[0]


def test_unobserved_gap_is_never_created_as_ground_or_used_as_bridge():
    grid, mantle, guard = _synthetic_scene()
    # Entire measured row missing; no PTD anchors on the far bank.
    grid.point_count.reshape(grid.ny, grid.nx)[12, :] = 0
    grid.geometry_count.reshape(grid.ny, grid.nx)[12:, :] = 0
    grid.strong_count.reshape(grid.ny, grid.nx)[12:, :] = 0
    mantle.observed[12, :] = False
    mantle.reliable[12:, :] = False
    continuity = build_ground_continuity(grid, mantle, guard)
    assert not np.any(continuity.connected[12, :])
    assert not np.any(continuity.connected[13:, :])
    x, y, z = _xyz(grid, 10, 12)
    assert not continuity.recovery_mask(x, y, z)[0]


def test_roof_candidate_blocks_graph_even_with_strong_source_class2_ptd():
    grid, mantle, guard = _synthetic_scene()
    guard.roof_candidate[12, 10] = True
    continuity = build_ground_continuity(grid, mantle, guard)
    x, y, z = _xyz(grid, 10, 12)
    assert continuity.blocked[12, 10]
    assert not continuity.connected[12, 10]
    assert not continuity.recovery_mask(x, y, z)[0]


def test_near_surface_return_is_recovered_but_elevated_canopy_is_not():
    grid, mantle, guard = _synthetic_scene()
    continuity = build_ground_continuity(grid, mantle, guard)
    x, y, low = _xyz(grid, 10, 12)
    high = low + 1.0
    assert continuity.recovery_mask(x, y, low)[0]
    assert not continuity.recovery_mask(x, y, high)[0]
    assert guard.classify_veto(x, y, high)[0] == VETO_HEIGHT
    assert not continuity.recovery_mask(
        x, y, low, veto_codes=np.array([VETO_ROOF_CANDIDATE], dtype=np.uint8)
    )[0]


def test_steep_face_uses_normal_not_global_vertical_height():
    grid, mantle, guard = _synthetic_scene(slope=1.35)
    continuity = build_ground_continuity(grid, mantle, guard)
    x, y, z = _xyz(grid, 10, 12)
    assert continuity.connected[12, 10]
    assert continuity.recovery_mask(x, y, z)[0]
    # Same true terrain plane at an offset within its raster cell.
    moved_x = x + 0.07
    moved_z = z + 1.35 * 0.07
    assert continuity.recovery_mask(moved_x, y, moved_z)[0]


def test_point_order_invariant_for_continuity_recovery():
    grid, mantle, guard = _synthetic_scene()
    continuity = build_ground_continuity(grid, mantle, guard)
    p0 = _xyz(grid, 10, 12)
    p1 = _xyz(grid, 11, 12)
    x = np.r_[p0[0], p1[0], p0[0]]
    y = np.r_[p0[1], p1[1], p0[1]]
    z = np.r_[p0[2], p1[2], p0[2] + 2.0]
    expected = continuity.recovery_mask(x, y, z)
    order = np.array([2, 0, 1], dtype=np.int64)
    np.testing.assert_array_equal(
        expected[order], continuity.recovery_mask(x[order], y[order], z[order])
    )


def test_post_veto_integration_has_provenance_and_no_reacceptance():
    grid, mantle, guard = _synthetic_scene()
    continuity = build_ground_continuity(grid, mantle, guard)
    a = _xyz(grid, 10, 12)
    b = _xyz(grid, 11, 12)
    x = np.array([a[0][0], a[0][0], b[0][0], a[0][0]])
    y = np.array([a[1][0], a[1][0], b[1][0], a[1][0]])
    z = np.array([a[2][0], a[2][0] + 1.0, b[2][0], a[2][0]])
    guard.roof_candidate[12, 11] = True
    cfg = GroundEvidenceConfig(
        surface_scale=0.20, detrend_scale=0.50,
        vertical_spread_limit=0.75, roughness_scale=0.20,
    )
    evidence = GroundEvidenceScorer(cfg).evaluate(
        ptd_score=np.full(4, 0.01),
        tin_residual=np.full(4, np.inf),
        vertical_residual=np.full(4, np.nan),
        detrended_residual=np.full(4, np.nan),
        neighbour_support=np.full(4, 0.15),
        vertical_spread=np.zeros(4), roughness=np.zeros(4),
        original_class=np.array([2, 2, 2, 2], dtype=np.uint8),
    )
    evidence.decision[:] = int(GroundDecision.UNKNOWN)
    applied = apply_continuity_recovery(
        evidence, continuity, guard, x, y, z,
        invalid=np.array([False, False, False, True]),
    )
    np.testing.assert_array_equal(
        applied, np.array([True, False, False, False])
    )
    assert evidence.decision[0] == GroundDecision.L3_GROUND_CONTINUITY_RECOVERED
    assert evidence.ground_source_codes()[0] == 6
    assert (evidence.provenance[0] & PROV_GROUND_CONTINUITY) != 0
    assert evidence.classifications().tolist() == [2, 1, 1, 1]
    _, reasons = diagnose_ground_gates(evidence, cfg)
    assert reasons[0] == GroundRejectReason.ACCEPTED
