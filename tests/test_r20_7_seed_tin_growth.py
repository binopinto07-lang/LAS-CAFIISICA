from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from las_classifier.terrain.seed_tin_growth import (
    SeedTINGrowthConfig,
    build_seed_tin_ground_guard,
)


def _grid_and_mantle(
    *,
    nx: int = 48,
    ny: int = 48,
    cell: float = 1.0,
    slope_x: float = 0.25,
    slope_y: float = 0.10,
):
    x, y = np.meshgrid(
        np.arange(nx, dtype=np.float64) * cell,
        np.arange(ny, dtype=np.float64) * cell,
    )
    lower = 100.0 + slope_x * x + slope_y * y
    upper = lower + 0.05
    shape = lower.shape
    grid = SimpleNamespace(
        origin=np.array([5000.0, 10000.0], dtype=np.float64),
        cell_size=cell,
        nx=nx,
        ny=ny,
        min_z=lower.astype(np.float32).ravel(),
        max_z=upper.astype(np.float32).ravel(),
        point_count=np.full(lower.size, 8, dtype=np.int32),
    )
    mantle = SimpleNamespace(
        origin=grid.origin.copy(),
        cell_size=cell,
        nx=nx,
        ny=ny,
        observed=np.ones(shape, dtype=np.bool_),
        reliable=np.ones(shape, dtype=np.bool_),
        slope_x=np.full(shape, slope_x, dtype=np.float32),
        slope_y=np.full(shape, slope_y, dtype=np.float32),
        breakline=np.zeros(shape, dtype=np.bool_),
        veto_guard=SimpleNamespace(
            roof_candidate=np.zeros(shape, dtype=np.bool_),
            canopy_candidate=np.zeros(shape, dtype=np.bool_),
        ),
    )
    return grid, mantle


def test_clean_inclined_terrain_is_not_blocked():
    grid, mantle = _grid_and_mantle(slope_x=0.8, slope_y=0.35)
    guard = build_seed_tin_ground_guard(
        grid,
        mantle,
        config=SeedTINGrowthConfig(seed_cell_m=8.0),
    )

    assert guard.seed_count >= 3
    assert guard.inside_tin.any()
    assert not guard.blocked.any()
    assert guard.accepted.sum() > guard.seed_count


def test_elevated_object_patch_above_low_seed_tin_is_blocked():
    grid, mantle = _grid_and_mantle()
    lower = grid.min_z.reshape(grid.ny, grid.nx).copy()
    upper = grid.max_z.reshape(grid.ny, grid.nx).copy()

    # The coarse seed blocks still contain lower terrain around this patch.
    lower[20:25, 20:25] += 1.40
    upper[20:25, 20:25] = lower[20:25, 20:25] + 0.15
    grid.min_z = lower.ravel()
    grid.max_z = upper.ravel()

    guard = build_seed_tin_ground_guard(
        grid,
        mantle,
        config=SeedTINGrowthConfig(seed_cell_m=10.0),
    )

    assert guard.blocked[22, 22]
    assert guard.normal_distance[22, 22] > 0.58


def test_breakline_is_never_hard_blocked_by_seed_tin_guard():
    grid, mantle = _grid_and_mantle()
    lower = grid.min_z.reshape(grid.ny, grid.nx).copy()
    lower[20:25, 20:25] += 1.50
    grid.min_z = lower.ravel()
    mantle.breakline[22, 22] = True

    guard = build_seed_tin_ground_guard(
        grid,
        mantle,
        config=SeedTINGrowthConfig(seed_cell_m=10.0),
    )

    assert not guard.blocked[22, 22]


def test_cells_below_coarse_tin_are_not_rejected():
    grid, mantle = _grid_and_mantle()
    lower = grid.min_z.reshape(grid.ny, grid.nx).copy()
    lower[22, 22] -= 1.5
    grid.min_z = lower.ravel()

    guard = build_seed_tin_ground_guard(
        grid,
        mantle,
        config=SeedTINGrowthConfig(seed_cell_m=8.0),
    )

    # A coarse TIN may bridge across a true depression. R20.7 never rejects
    # cells merely for falling below that TIN.
    assert not guard.blocked[22, 22]


def test_seed_selection_is_not_based_on_original_class_or_returns():
    source = __import__(
        "pathlib"
    ).Path("src/las_classifier/terrain/seed_tin_growth.py").read_text(
        encoding="utf-8"
    )
    assert "class2" not in source.lower()
    assert "return_number" not in source
    assert "number_of_returns" not in source


def test_object_veto_cells_cannot_become_low_seeds():
    grid, mantle = _grid_and_mantle()
    extra = np.zeros((grid.ny, grid.nx), dtype=bool)
    extra[:8, :8] = True

    guard = build_seed_tin_ground_guard(
        grid,
        mantle,
        config=SeedTINGrowthConfig(seed_cell_m=8.0),
        extra_blocked=extra,
    )

    assert not guard.seed_cells[:8, :8].any()
    assert guard.seed_count >= 3
