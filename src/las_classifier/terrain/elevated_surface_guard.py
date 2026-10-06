"""R20.5 multiscale geometric veto for elevated islands in universal LAS Ground.

Runs on ALL sources (P1/L3/unknown). It never reads original LAS class labels,
sensor names or return count. Only measured cell envelopes and local geometry
are allowed to veto; the inferred mantle is NEVER sole proof of real Ground.

A terrace face is protected where the existing breakline detector is uncertain.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np

LOGGER = logging.getLogger("las_cafiisica.terrain.elevated_surface_guard")

_DIRECTIONS = (
    (-1, -1), (-1, 0), (-1, 1),
    (0, -1), (0, 1),
    (1, -1), (1, 0), (1, 1),
)
_OPPOSITE_PAIRS = ((0, 7), (1, 6), (2, 5), (3, 4))


@dataclass(frozen=True, slots=True)
class ElevatedGuardConfig:
    min_support_returns: int = 3
    min_prominence_normal_m: float = 0.65
    min_support_rays: int = 3
    distances_m: tuple[float, ...] = (2.0, 4.0, 7.0)
    max_reference_gradient: float = 3.0
    max_candidate_cell_spread_m: float = 2.75


def _shift(array: np.ndarray, dy: int, dx: int, fill: float) -> np.ndarray:
    """Sample array[y+dy, x+dx]; missing support stays invalid, never wraps."""
    h, w = array.shape
    result = np.full((h, w), fill, dtype=array.dtype)
    if abs(dy) >= h or abs(dx) >= w:
        return result
    y0, y1 = max(0, -dy), h - max(0, dy)
    x0, x1 = max(0, -dx), w - max(0, dx)
    result[y0:y1, x0:x1] = array[y0 + dy:y1 + dy, x0 + dx:x1 + dx]
    return result


def elevated_island_mask(grid, mantle, config: ElevatedGuardConfig | None = None) -> np.ndarray:
    """Return a conservative cell veto for suspended roof/shrub/rock islands.

    For a candidate cell, compare its physically measured LOW envelope with
    measured neighbour envelopes extrapolated by their *local tangent planes*.
    A true planar talude therefore has near-zero normal residual even where
    global Z changes rapidly. A floating object may sit on the local mantle,
    but stands above terrain predicted independently from around its edges.

    At least three rays plus one opposite-direction pair are required. This
    prevents a normal one-sided terrace step/downhill face being marked as an
    elevated island. No synthetic samples, interpolation or class-2 priors.
    """
    cfg = config or ElevatedGuardConfig()
    shape = (int(grid.ny), int(grid.nx))
    observed = np.asarray(mantle.observed, dtype=np.bool_)
    if observed.shape != shape:
        raise ValueError("R20.5 evidence / mantle shape mismatch")
    lower = np.asarray(grid.min_z, dtype=np.float32).reshape(shape)
    upper = np.asarray(grid.max_z, dtype=np.float32).reshape(shape)
    counts = np.asarray(grid.point_count).reshape(shape)
    reliable = (
        observed
        & (counts >= cfg.min_support_returns)
        & np.isfinite(lower)
    )
    sx = np.asarray(mantle.slope_x, dtype=np.float32)
    sy = np.asarray(mantle.slope_y, dtype=np.float32)
    if sx.shape != shape or sy.shape != shape:
        raise ValueError("R20.5 slope-grid shape mismatch")

    candidate = reliable & (upper - lower <= cfg.max_candidate_cell_spread_m)
    breakline = getattr(mantle, "breakline", None)
    if breakline is not None:
        candidate &= ~np.asarray(breakline, dtype=np.bool_).reshape(shape)

    cell = float(grid.cell_size)
    offsets = sorted(set(max(1, int(round(m / cell))) for m in cfg.distances_m))
    ray_votes = np.zeros(shape, dtype=np.uint8)
    pair_flags: list[np.ndarray] = []

    for dy, dx in _DIRECTIONS:
        # The best witness along this ray must be physically observed; an
        # inferred mantle cell is deliberately not considered a witness.
        ray = np.zeros(shape, dtype=np.bool_)
        for n in offsets:
            offset_y, offset_x = dy * n, dx * n
            neighbour_z = _shift(lower, offset_y, offset_x, np.float32(np.nan))
            neighbour_sx = _shift(sx, offset_y, offset_x, np.float32(np.nan))
            neighbour_sy = _shift(sy, offset_y, offset_x, np.float32(np.nan))
            near_measured = _shift(reliable, offset_y, offset_x, False)
            projection = (
                neighbour_z
                - neighbour_sx * (dx * n * cell)
                - neighbour_sy * (dy * n * cell)
            )
            slope_scale = np.sqrt(
                1.0 + neighbour_sx * neighbour_sx + neighbour_sy * neighbour_sy
            )
            normal_prominence = (lower - projection) / np.maximum(slope_scale, 1.0)
            ray |= (
                candidate
                & near_measured
                & np.isfinite(normal_prominence)
                & (np.hypot(neighbour_sx, neighbour_sy) <= cfg.max_reference_gradient)
                & (normal_prominence >= cfg.min_prominence_normal_m)
            )
        pair_flags.append(ray)
        ray_votes += ray.astype(np.uint8)

    opposed = np.zeros(shape, dtype=np.bool_)
    for a, b in _OPPOSITE_PAIRS:
        opposed |= pair_flags[a] & pair_flags[b]

    blocked = candidate & (ray_votes >= cfg.min_support_rays) & opposed
    LOGGER.info(
        "R20_5_ELEVATED_ISLAND candidate=%d blocked=%d measured_only=1",
        int(candidate.sum()),
        int(blocked.sum()),
    )
    return blocked
