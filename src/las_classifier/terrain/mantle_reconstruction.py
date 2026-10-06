"""R20.6 reconstruction of NO_GROUND_OBSERVATION from the preserved mantle.

This is NOT measured Ground. It creates synthetic class-2 support only in
mantle cells explicitly tagged as inferred/no-Ground-observation. Measured
points, the R20 mantle, and its veto masks are never modified.
"""
from __future__ import annotations

from dataclasses import dataclass
from math import ceil, floor, sqrt
from typing import Iterator

import numpy as np
from scipy.ndimage import distance_transform_edt


@dataclass(frozen=True, slots=True)
class MantleReconstructionConfig:
    spacing_m: float = 0.25
    max_points: int = 12_000_000
    max_anchor_distance_m: float = 3.0
    max_hidden_projection_delta_m: float = 3.0


@dataclass(slots=True)
class MantleGroundReconstruction:
    origin: np.ndarray
    cell_size: float
    nx: int
    ny: int
    surface: np.ndarray
    slope_x: np.ndarray
    slope_y: np.ndarray
    fill_mask: np.ndarray
    source_state: np.ndarray
    config: MantleReconstructionConfig

    @property
    def fill_cell_count(self) -> int:
        return int(np.count_nonzero(self.fill_mask))

    @property
    def fill_divisions(self) -> int:
        cells = self.fill_cell_count
        if cells <= 0:
            return 0
        requested = max(
            1,
            int(ceil(self.cell_size / max(self.config.spacing_m, 0.01))),
        )
        permitted = max(
            1,
            int(floor(sqrt(max(1, self.config.max_points) / cells))),
        )
        return min(requested, permitted)

    @property
    def effective_spacing(self) -> float:
        div = self.fill_divisions
        return self.cell_size / div if div > 0 else float(self.config.spacing_m)

    @property
    def point_count(self) -> int:
        div = self.fill_divisions
        return self.fill_cell_count * div * div

    @property
    def inferred_cell_count(self) -> int:
        return int(np.count_nonzero(self.source_state == 1))

    @property
    def hidden_ground_cell_count(self) -> int:
        return int(np.count_nonzero(self.source_state == 2))

    @property
    def reliable_missing_cell_count(self) -> int:
        return int(np.count_nonzero(self.source_state == 3))

    def iter_xyz(
        self,
        chunk_points: int = 500_000,
    ) -> Iterator[tuple[np.ndarray, np.ndarray, np.ndarray]]:
        """Generate bounded synthetic XYZ chunks on the reconstruction surface."""
        div = self.fill_divisions
        if div <= 0:
            return

        rows, cols = np.nonzero(self.fill_mask)
        per_cell = div * div
        cells_per_chunk = max(1, int(chunk_points) // per_cell)
        offsets = (np.arange(div, dtype=np.float64) + 0.5) / div
        off_x, off_y = np.meshgrid(offsets, offsets)
        off_x = off_x.ravel()
        off_y = off_y.ravel()

        for start in range(0, rows.size, cells_per_chunk):
            stop = min(start + cells_per_chunk, rows.size)
            rr_i = rows[start:stop]
            cc_i = cols[start:stop]
            rr = rr_i.astype(np.float64, copy=False)
            cc = cc_i.astype(np.float64, copy=False)

            local_x = np.broadcast_to(off_x, (rr.size, per_cell)).ravel()
            local_y = np.broadcast_to(off_y, (rr.size, per_cell)).ravel()
            gx = (cc[:, None] + off_x[None, :]).ravel()
            gy = (rr[:, None] + off_y[None, :]).ravel()
            x = self.origin[0] + gx * self.cell_size
            y = self.origin[1] + gy * self.cell_size

            centre_z = np.repeat(
                self.surface[rr_i, cc_i].astype(np.float64), per_cell
            )
            sx = np.repeat(
                self.slope_x[rr_i, cc_i].astype(np.float64), per_cell
            )
            sy = np.repeat(
                self.slope_y[rr_i, cc_i].astype(np.float64), per_cell
            )
            dx = (local_x - 0.5) * self.cell_size
            dy = (local_y - 0.5) * self.cell_size
            z = centre_z + sx * dx + sy * dy
            finite = np.isfinite(x) & np.isfinite(y) & np.isfinite(z)
            if np.any(finite):
                yield x[finite], y[finite], z[finite]


def build_mantle_ground_reconstruction(
    mantle,
    *,
    measured_ground_cells: np.ndarray | None = None,
    spacing_m: float = 0.25,
    max_points: int = 12_000_000,
) -> MantleGroundReconstruction:
    """Build synthetic Ground only from explicit non-observation mantle states.

    State 1: empty/unobserved cell inferred by the R20 mantle.
    State 2: measured upper returns exist, but Ground itself is not observed.
    State 3: mantle is reliable, but no measured point survived FINAL GROUND.

    For state 2, the raw mantle can ride the visible canopy/object. Therefore
    the reconstruction centre is projected from the nearest reliable measured
    Ground anchor instead of copying an upper visible return.
    """
    shape = (int(mantle.ny), int(mantle.nx))
    surface = np.asarray(mantle.surface, dtype=np.float32)
    if surface.shape != shape:
        raise ValueError("R20.6 mantle surface shape mismatch")

    inferred = np.asarray(mantle.inferred, dtype=np.bool_)
    hidden = np.asarray(
        mantle.possible_no_ground_observation, dtype=np.bool_
    )
    reliable = np.asarray(mantle.reliable, dtype=np.bool_)
    observed = np.asarray(mantle.observed, dtype=np.bool_)
    if (
        inferred.shape != shape
        or hidden.shape != shape
        or reliable.shape != shape
        or observed.shape != shape
    ):
        raise ValueError("R20.6 mantle state shape mismatch")

    if measured_ground_cells is None:
        measured_ground = reliable.copy()
    else:
        measured_ground = np.asarray(
            measured_ground_cells, dtype=np.bool_
        )
        if measured_ground.shape != shape:
            raise ValueError("R20.6 measured-Ground cell shape mismatch")
    reliable_missing = reliable & ~measured_ground

    breakline = getattr(mantle, "breakline", None)
    if breakline is None:
        breakline = np.zeros(shape, dtype=np.bool_)
    else:
        breakline = np.asarray(breakline, dtype=np.bool_).reshape(shape)

    guard = getattr(mantle, "veto_guard", None)
    elevated = np.zeros(shape, dtype=np.bool_)
    if guard is not None:
        elevated = (
            np.asarray(guard.roof_candidate, dtype=np.bool_)
            | np.asarray(guard.canopy_candidate, dtype=np.bool_)
        )
        if elevated.shape != shape:
            raise ValueError("R20.6 elevated-candidate shape mismatch")

    # If the visible surface itself was flagged as elevated and no accepted
    # measured Ground survived in that cell, treat it as HIDDEN GROUND. Never
    # copy the roof/canopy Z into the reconstructed Ground.
    hidden = hidden | (observed & ~measured_ground & elevated)
    reliable_missing = reliable & ~measured_ground & ~elevated

    anchor = (
        reliable
        & measured_ground
        & np.isfinite(surface)
        & ~breakline
        & ~elevated
    )

    config = MantleReconstructionConfig(
        spacing_m=max(0.05, float(spacing_m)),
        max_points=max(1, int(max_points)),
    )

    fill = (
        inferred | hidden | reliable_missing
    ) & np.isfinite(surface) & ~breakline
    out_surface = surface.copy()
    out_sx = np.asarray(mantle.slope_x, dtype=np.float32).copy()
    out_sy = np.asarray(mantle.slope_y, dtype=np.float32).copy()

    if np.any(fill) and np.any(anchor):
        distance, nearest = distance_transform_edt(
            ~anchor, return_indices=True
        )
        allowed = (
            distance * float(mantle.cell_size)
            <= config.max_anchor_distance_m
        )
        fill &= allowed

        # Hidden Ground under visible upper returns must be estimated from a
        # nearby reliable terrain tangent plane, not from the canopy/object Z.
        hidden_fill = hidden & fill
        rr, cc = np.nonzero(hidden_fill)
        if rr.size:
            ar = nearest[0, rr, cc]
            ac = nearest[1, rr, cc]
            anchor_z = surface[ar, ac].astype(np.float64)
            anchor_sx = np.asarray(mantle.slope_x, dtype=np.float32)[ar, ac].astype(np.float64)
            anchor_sy = np.asarray(mantle.slope_y, dtype=np.float32)[ar, ac].astype(np.float64)
            dx = (cc - ac).astype(np.float64) * float(mantle.cell_size)
            dy = (rr - ar).astype(np.float64) * float(mantle.cell_size)
            projected = anchor_z + anchor_sx * dx + anchor_sy * dy
            raw = surface[rr, cc].astype(np.float64)
            projected = np.clip(
                projected,
                raw - config.max_hidden_projection_delta_m,
                raw + 0.50,
            )
            out_surface[rr, cc] = projected.astype(np.float32)
            out_sx[rr, cc] = anchor_sx.astype(np.float32)
            out_sy[rr, cc] = anchor_sy.astype(np.float32)
    elif np.any(fill):
        # No trustworthy Ground anchor means no synthetic Ground authority.
        fill[:] = False

    state = np.zeros(shape, dtype=np.uint8)
    state[inferred & fill] = 1
    state[hidden & fill] = 2
    state[reliable_missing & fill] = 3

    return MantleGroundReconstruction(
        origin=np.asarray(mantle.origin, dtype=np.float64).copy(),
        cell_size=float(mantle.cell_size),
        nx=int(mantle.nx),
        ny=int(mantle.ny),
        surface=out_surface,
        slope_x=out_sx,
        slope_y=out_sy,
        fill_mask=fill,
        source_state=state,
        config=config,
    )
