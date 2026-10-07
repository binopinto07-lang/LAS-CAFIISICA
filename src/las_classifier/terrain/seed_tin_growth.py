"""R20.7 conservative LOW-SEED GRID -> TIN -> iterative Ground growth.

Clean-room implementation inspired by the documented *principle* of progressive
terrain classification. It does not copy Agisoft code and never uses original
LAS class 2 as terrain authority.

The output is a conservative CELL veto. The preserved R20 mantle remains
read-only. Cells well above a low-seed terrain TIN are blocked from measured
Ground and later reconstructed, when justified, through the existing
NO_GROUND_OBSERVATION path.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from math import cos, radians

import numpy as np
from scipy.spatial import Delaunay, QhullError

LOGGER = logging.getLogger("las_cafiisica.terrain.seed_tin_growth")

_NEIGHBOURS = (
    (-1, -1), (-1, 0), (-1, 1),
    (0, -1),           (0, 1),
    (1, -1),  (1, 0),  (1, 1),
)


@dataclass(frozen=True, slots=True)
class SeedTINGrowthConfig:
    seed_cell_m: float = 12.0
    max_seed_points: int = 80_000
    min_returns: int = 2
    initial_distance_normal_m: float = 0.32
    growth_distance_normal_m: float = 0.48
    block_above_normal_m: float = 0.58
    max_angle_deg: float = 38.0
    max_iterations: int = 24
    min_growth_neighbours: int = 1


@dataclass(frozen=True, slots=True)
class SeedTINGroundGuard:
    origin: np.ndarray
    cell_size: float
    nx: int
    ny: int
    seed_cells: np.ndarray
    accepted: np.ndarray
    blocked: np.ndarray
    inside_tin: np.ndarray
    normal_distance: np.ndarray
    seed_count: int
    expansion_steps: int

    @property
    def blocked_cell_count(self) -> int:
        return int(np.count_nonzero(self.blocked))


def _shift(mask: np.ndarray, dy: int, dx: int) -> np.ndarray:
    out = np.zeros(mask.shape, dtype=np.bool_)
    h, w = mask.shape
    y0, y1 = max(0, -dy), h - max(0, dy)
    x0, x1 = max(0, -dx), w - max(0, dx)
    if y0 >= y1 or x0 >= x1:
        return out
    out[y0:y1, x0:x1] = mask[
        y0 + dy:y1 + dy,
        x0 + dx:x1 + dx,
    ]
    return out


def _seed_indices(
    lower: np.ndarray,
    valid: np.ndarray,
    *,
    cell_size: float,
    seed_cell_m: float,
    max_seed_points: int,
) -> tuple[np.ndarray, np.ndarray, int]:
    """Choose the lowest measured dense-grid cell in each coarse seed block."""
    ny, nx = lower.shape
    stride = max(1, int(round(seed_cell_m / cell_size)))
    max_seed_points = max(3, int(max_seed_points))

    while True:
        rr, cc = np.nonzero(valid)
        if rr.size == 0:
            return rr, cc, stride
        blocks_x = (nx + stride - 1) // stride
        block_id = (rr // stride) * blocks_x + (cc // stride)
        order = np.lexsort((lower[rr, cc], block_id))
        ordered_blocks = block_id[order]
        first = np.empty(order.size, dtype=np.bool_)
        first[0] = True
        first[1:] = ordered_blocks[1:] != ordered_blocks[:-1]
        chosen = order[first]
        if chosen.size <= max_seed_points:
            return rr[chosen], cc[chosen], stride
        stride = max(stride + 1, int(np.ceil(stride * 1.25)))


def _triangle_gradients(
    xy: np.ndarray,
    z: np.ndarray,
    simplices: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    tri = xy[simplices]
    zz = z[simplices]
    x1, y1 = tri[:, 0, 0], tri[:, 0, 1]
    x2, y2 = tri[:, 1, 0], tri[:, 1, 1]
    x3, y3 = tri[:, 2, 0], tri[:, 2, 1]
    z1, z2, z3 = zz[:, 0], zz[:, 1], zz[:, 2]
    det = (x2 - x1) * (y3 - y1) - (x3 - x1) * (y2 - y1)
    safe = np.where(np.abs(det) > 1e-12, det, np.nan)
    gx = ((z2 - z1) * (y3 - y1) - (z3 - z1) * (y2 - y1)) / safe
    gy = ((x2 - x1) * (z3 - z1) - (x3 - x1) * (z2 - z1)) / safe
    return gx.astype(np.float64), gy.astype(np.float64)


def build_seed_tin_ground_guard(
    grid,
    mantle,
    *,
    config: SeedTINGrowthConfig | None = None,
) -> SeedTINGroundGuard:
    cfg = config or SeedTINGrowthConfig()
    shape = (int(grid.ny), int(grid.nx))
    lower = np.asarray(grid.min_z, dtype=np.float64).reshape(shape)
    counts = np.asarray(grid.point_count).reshape(shape)
    observed = np.asarray(mantle.observed, dtype=np.bool_).reshape(shape)
    reliable = np.asarray(mantle.reliable, dtype=np.bool_).reshape(shape)
    sx = np.asarray(mantle.slope_x, dtype=np.float64).reshape(shape)
    sy = np.asarray(mantle.slope_y, dtype=np.float64).reshape(shape)
    breakline = getattr(mantle, "breakline", None)
    breakline = (
        np.zeros(shape, dtype=np.bool_)
        if breakline is None
        else np.asarray(breakline, dtype=np.bool_).reshape(shape)
    )

    veto = getattr(mantle, "veto_guard", None)
    existing_blocked = np.zeros(shape, dtype=np.bool_)
    if veto is not None:
        existing_blocked = (
            np.asarray(veto.roof_candidate, dtype=np.bool_)
            | np.asarray(veto.canopy_candidate, dtype=np.bool_)
        )

    valid_seed = (
        observed
        & np.isfinite(lower)
        & (counts >= cfg.min_returns)
        & ~existing_blocked
        & ~breakline
    )
    seed_r, seed_c, _ = _seed_indices(
        lower,
        valid_seed,
        cell_size=float(grid.cell_size),
        seed_cell_m=cfg.seed_cell_m,
        max_seed_points=cfg.max_seed_points,
    )

    seed_mask = np.zeros(shape, dtype=np.bool_)
    inside_tin = np.zeros(shape, dtype=np.bool_)
    accepted = np.zeros(shape, dtype=np.bool_)
    blocked = np.zeros(shape, dtype=np.bool_)
    normal_distance = np.full(shape, np.nan, dtype=np.float32)

    if seed_r.size < 3:
        return SeedTINGroundGuard(
            origin=np.asarray(grid.origin, dtype=np.float64).copy(),
            cell_size=float(grid.cell_size),
            nx=int(grid.nx), ny=int(grid.ny),
            seed_cells=seed_mask, accepted=accepted, blocked=blocked,
            inside_tin=inside_tin, normal_distance=normal_distance,
            seed_count=int(seed_r.size), expansion_steps=0,
        )

    seed_mask[seed_r, seed_c] = True
    seed_xy = np.column_stack((
        grid.origin[0] + (seed_c.astype(np.float64) + 0.5) * grid.cell_size,
        grid.origin[1] + (seed_r.astype(np.float64) + 0.5) * grid.cell_size,
    ))
    seed_z = lower[seed_r, seed_c]

    try:
        tin = Delaunay(seed_xy)
    except QhullError:
        LOGGER.warning("R20_7_TIN qhull failed; guard disabled")
        return SeedTINGroundGuard(
            origin=np.asarray(grid.origin, dtype=np.float64).copy(),
            cell_size=float(grid.cell_size),
            nx=int(grid.nx), ny=int(grid.ny),
            seed_cells=seed_mask, accepted=accepted, blocked=blocked,
            inside_tin=inside_tin, normal_distance=normal_distance,
            seed_count=int(seed_r.size), expansion_steps=0,
        )

    rr, cc = np.nonzero(observed & np.isfinite(lower))
    query_xy = np.column_stack((
        grid.origin[0] + (cc.astype(np.float64) + 0.5) * grid.cell_size,
        grid.origin[1] + (rr.astype(np.float64) + 0.5) * grid.cell_size,
    ))
    simplex = tin.find_simplex(query_xy)
    inside = simplex >= 0
    if not np.any(inside):
        return SeedTINGroundGuard(
            origin=np.asarray(grid.origin, dtype=np.float64).copy(),
            cell_size=float(grid.cell_size),
            nx=int(grid.nx), ny=int(grid.ny),
            seed_cells=seed_mask, accepted=accepted, blocked=blocked,
            inside_tin=inside_tin, normal_distance=normal_distance,
            seed_count=int(seed_r.size), expansion_steps=0,
        )

    qidx = np.flatnonzero(inside)
    simp = simplex[inside]
    qxy = query_xy[inside]
    transform = tin.transform[simp]
    bary2 = np.einsum(
        "nij,nj->ni",
        transform[:, :2, :],
        qxy - transform[:, 2, :],
    )
    bary = np.column_stack((bary2, 1.0 - bary2.sum(axis=1)))
    vertices = tin.simplices[simp]
    zref = np.sum(seed_z[vertices] * bary, axis=1)

    gx_all, gy_all = _triangle_gradients(seed_xy, seed_z, tin.simplices)
    gx = gx_all[simp]
    gy = gy_all[simp]
    norm = np.sqrt(1.0 + gx * gx + gy * gy)
    residual = (lower[rr[inside], cc[inside]] - zref) / np.maximum(norm, 1.0)

    inside_tin[rr[inside], cc[inside]] = True
    normal_distance[rr[inside], cc[inside]] = residual.astype(np.float32)

    mantle_gx = sx[rr[inside], cc[inside]]
    mantle_gy = sy[rr[inside], cc[inside]]
    n1 = np.column_stack((-gx, -gy, np.ones_like(gx)))
    n2 = np.column_stack((-mantle_gx, -mantle_gy, np.ones_like(mantle_gx)))
    cosine = np.sum(n1 * n2, axis=1) / (
        np.linalg.norm(n1, axis=1) * np.linalg.norm(n2, axis=1)
    )
    angle_ok_values = cosine >= cos(radians(cfg.max_angle_deg))

    angle_ok = np.zeros(shape, dtype=np.bool_)
    angle_ok[rr[inside], cc[inside]] = angle_ok_values

    compatible = (
        inside_tin
        & observed
        & ~existing_blocked
        & ~breakline
        & angle_ok
        & np.isfinite(normal_distance)
        & (np.abs(normal_distance) <= cfg.growth_distance_normal_m)
    )
    initial = (
        compatible
        & (np.abs(normal_distance) <= cfg.initial_distance_normal_m)
    )
    accepted = seed_mask | initial
    steps = 0
    for iteration in range(max(1, int(cfg.max_iterations))):
        votes = np.zeros(shape, dtype=np.uint8)
        for dy, dx in _NEIGHBOURS:
            votes += _shift(accepted, dy, dx).astype(np.uint8)
        update = (
            compatible
            & ~accepted
            & (votes >= cfg.min_growth_neighbours)
        )
        if not np.any(update):
            break
        accepted |= update
        steps = iteration + 1

    # Only cells CLEARLY ABOVE the low-seed terrain are hard-blocked.
    # Cells below the TIN are never rejected by this guard because the initial
    # coarse TIN may bridge across a real valley or terrace transition.
    blocked = (
        inside_tin
        & observed
        & ~breakline
        & np.isfinite(normal_distance)
        & (normal_distance > cfg.block_above_normal_m)
        & ~accepted
    )

    LOGGER.info(
        "R20_7_SEED_TIN seeds=%d inside=%d accepted=%d blocked=%d passes=%d",
        int(seed_r.size),
        int(np.count_nonzero(inside_tin)),
        int(np.count_nonzero(accepted)),
        int(np.count_nonzero(blocked)),
        int(steps),
    )
    return SeedTINGroundGuard(
        origin=np.asarray(grid.origin, dtype=np.float64).copy(),
        cell_size=float(grid.cell_size),
        nx=int(grid.nx), ny=int(grid.ny),
        seed_cells=seed_mask,
        accepted=accepted,
        blocked=blocked,
        inside_tin=inside_tin,
        normal_distance=normal_distance,
        seed_count=int(seed_r.size),
        expansion_steps=int(steps),
    )
