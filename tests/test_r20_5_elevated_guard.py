"""Geometric R20.5 veto regressions. No source class-2 / sensor shortcuts."""
from types import SimpleNamespace

import numpy as np

from las_classifier.terrain.elevated_surface_guard import elevated_island_mask, vertical_structure_mask


def make_evidence(
    *,
    slope_x: float = 0.4,
    slope_y: float = 0.1,
    isolated_roof: bool = False,
    terrace: bool = False,
):
    cell = 0.5
    x, y = np.meshgrid(np.arange(50, dtype=np.float32) * cell,
                       np.arange(50, dtype=np.float32) * cell)
    z = 10.0 + slope_x * x + slope_y * y
    if isolated_roof:
        z[20:29, 20:29] += 1.8
    if terrace:
        z[:, 25:] += 1.9
    shape = z.shape
    grid = SimpleNamespace(
        nx=shape[1],
        ny=shape[0],
        cell_size=cell,
        min_z=z.copy().ravel(),
        max_z=(z + 0.08).ravel(),
        point_count=np.full(z.size, 10, dtype=np.uint32),
    )
    mantle = SimpleNamespace(
        observed=np.ones(shape, dtype=bool),
        slope_x=np.full(shape, slope_x, dtype=np.float32),
        slope_y=np.full(shape, slope_y, dtype=np.float32),
        breakline=np.zeros(shape, dtype=bool),
    )
    return grid, mantle


def test_roof_riding_terrain_surface_is_blocked():
    grid, mantle = make_evidence(isolated_roof=True)
    mask = elevated_island_mask(grid, mantle)
    assert mask.dtype == np.bool_
    assert mask[24, 24]
    assert not mask[2, 2]


def test_real_steep_talude_plane_is_not_blocked():
    grid, mantle = make_evidence(slope_x=1.6, slope_y=0.5)
    assert not np.any(elevated_island_mask(grid, mantle))


def test_one_sided_terrace_step_is_not_a_suspended_island():
    grid, mantle = make_evidence(slope_x=0, slope_y=0, terrace=True)
    assert not np.any(elevated_island_mask(grid, mantle))


def test_real_breaklines_remain_protected():
    grid, mantle = make_evidence(isolated_roof=True)
    mantle.breakline[20:29, 20:29] = True
    mask = elevated_island_mask(grid, mantle)
    assert not mask[24, 24]


def test_invalid_evidence_shape_is_rejected():
    grid, mantle = make_evidence()
    mantle.observed = np.ones((3, 3), dtype=bool)
    import pytest
    with pytest.raises(ValueError, match="shape"):
        elevated_island_mask(grid, mantle)


def test_vertical_structure_rejects_thick_irregular_p1_clutter():
    grid, mantle = make_evidence(slope_x=0.4, slope_y=0.1)
    shape = (grid.ny, grid.nx)
    upper = np.asarray(grid.max_z, dtype=np.float32).reshape(shape)
    lower = np.asarray(grid.min_z, dtype=np.float32).reshape(shape)
    # Simulate a dense vegetation/object patch with strong normal thickness.
    upper[20:30, 20:30] = lower[20:30, 20:30] + 0.95
    # Disturb the low envelope so the patch disagrees with neighbouring planes.
    lower[23:27, 23:27] += 0.35
    grid.min_z = lower.ravel()
    grid.max_z = upper.ravel()

    mask = vertical_structure_mask(grid, mantle)
    assert mask[24, 24]
    assert not mask[2, 2]


def test_vertical_structure_does_not_reject_clean_steep_plane():
    grid, mantle = make_evidence(slope_x=2.0, slope_y=0.7)
    mask = vertical_structure_mask(grid, mantle)
    assert not np.any(mask)
