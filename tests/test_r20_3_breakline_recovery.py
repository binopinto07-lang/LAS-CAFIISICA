"""R20.3 regression: measured terrace-face cells can be recovered without bridging a breakline."""
from __future__ import annotations

import numpy as np

from las_classifier.terrain.dense_spatial_evidence import DenseSpatialEvidenceGrid
from las_classifier.terrain.ground_continuity import build_ground_continuity
from las_classifier.terrain.inverted_mantle import InvertedGroundMantle, MantleConfig
from las_classifier.terrain.mantle_veto import MantleVeto, MantleVetoConfig, VETO_NONE


def _synthetic_scene():
    """Self-contained fixture: do not depend on another test module."""
    grid = DenseSpatialEvidenceGrid.create(
        np.array([0.0, 0.0]), np.array([11.0, 11.0]),
        spacing=0.05, strong_threshold=0.68, max_cells=10_000,
    )
    ny, nx = grid.ny, grid.nx
    yy, xx = np.indices((ny, nx))
    lower = 10.0 + 0.0 * (xx + 0.5) * grid.cell_size
    grid.point_count[:] = 4
    grid.geometry_count[:] = 4
    grid.strong_count[:] = 3
    grid.min_z[:] = lower.ravel()
    grid.max_z[:] = (lower + 0.04).ravel()

    observed = np.ones((ny, nx), dtype=np.bool_)
    reliable = np.ones((ny, nx), dtype=np.bool_)
    reliable[12, 3:-3] = False
    grid.geometry_count.reshape(ny, nx)[12, 3:-3] = 0
    grid.strong_count.reshape(ny, nx)[12, 3:-3] = 0

    surface = lower.astype(np.float32)
    gx = np.zeros((ny, nx), dtype=np.float32)
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


def _xyz(grid, ix, iy):
    x = np.array([(ix + 0.5) * grid.cell_size], dtype=np.float64)
    y = np.array([(iy + 0.5) * grid.cell_size], dtype=np.float64)
    z = float(grid.min_z[ix + iy * grid.nx])
    return x, y, np.array([z], dtype=np.float64)


def test_breakline_cell_is_recoverable_as_terminal_face():
    grid, mantle, guard = _synthetic_scene()
    mantle.breakline = np.zeros((grid.ny, grid.nx), dtype=np.bool_)
    mantle.breakline[12, 10] = True

    continuity = build_ground_continuity(grid, mantle, guard)

    x, y, z = _xyz(grid, 10, 12)
    assert continuity.breakline[12, 10]
    assert continuity.connected[12, 10]
    assert continuity.recovery_mask(
        x, y, z, veto_codes=np.array([VETO_NONE], dtype=np.uint8)
    )[0]


def test_breakline_cell_cannot_bridge_to_opposite_terrace():
    grid, mantle, guard = _synthetic_scene()
    mantle.breakline = np.zeros((grid.ny, grid.nx), dtype=np.bool_)
    mantle.breakline[12, 10] = True

    continuity = build_ground_continuity(grid, mantle, guard)

    # The terminal breakline cell itself may be recovered, but the graph must
    # not use it as a propagation source for the cell on the opposite side.
    assert continuity.connected[12, 10]
    assert not continuity.source_anchors[12, 10]
