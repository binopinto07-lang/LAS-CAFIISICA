from __future__ import annotations

import logging
from dataclasses import dataclass
from functools import lru_cache
from math import ceil
from time import perf_counter
from typing import Callable

import numpy as np
from numpy.typing import NDArray
from scipy.ndimage import (
    convolve,
    distance_transform_edt,
    grey_opening,
    map_coordinates,
)

from ..cloud.model import CloudModel


LOGGER = logging.getLogger("las_cafiisica.classifiers.smrf")

GROUND_CLASS = np.uint8(2)
NON_GROUND_CLASS = np.uint8(1)
ProgressCallback = Callable[[int, str], None]


@dataclass(frozen=True, slots=True)
class SMRFParams:
    cell: float = 1.0
    slope: float = 0.15
    window: float = 18.0
    threshold: float = 0.5
    scalar: float = 1.25
    chunk_size: int = 2_000_000
    max_grid_cells: int = 8_000_000
    inpaint_iterations: int = 250
    inpaint_tolerance: float = 0.001

    def validate(self) -> None:
        if self.cell <= 0:
            raise ValueError("SMRF cell must be greater than zero")
        if self.slope < 0:
            raise ValueError("SMRF slope cannot be negative")
        if self.window <= 0:
            raise ValueError("SMRF window must be greater than zero")
        if self.threshold < 0:
            raise ValueError("SMRF threshold cannot be negative")
        if self.scalar < 0:
            raise ValueError("SMRF scalar cannot be negative")
        if self.chunk_size < 1:
            raise ValueError("SMRF chunk_size must be positive")
        if self.max_grid_cells < 1:
            raise ValueError("SMRF max_grid_cells must be positive")
        if self.inpaint_iterations < 1:
            raise ValueError("SMRF inpaint_iterations must be positive")
        if self.inpaint_tolerance <= 0:
            raise ValueError("SMRF inpaint_tolerance must be positive")


@dataclass(frozen=True, slots=True)
class SMRFModel:
    params: SMRFParams
    min_x: float
    min_y: float
    rows: int
    cols: int
    ground_surface: NDArray[np.float32]
    slope_surface: NDArray[np.float32]
    object_cell_mask: NDArray[np.bool_]
    inpaint_cell_mask: NDArray[np.bool_]

    def _grid_coordinates(
        self,
        x: NDArray[np.float64],
        y: NDArray[np.float64],
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        cols = (x - self.min_x) / self.params.cell
        rows = (y - self.min_y) / self.params.cell
        np.clip(cols, 0.0, self.cols - 1.0, out=cols)
        np.clip(rows, 0.0, self.rows - 1.0, out=rows)
        return rows, cols

    def _sample(
        self,
        surface: NDArray[np.float32],
        rows: NDArray[np.float64],
        cols: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        return map_coordinates(
            surface,
            [rows, cols],
            order=1,
            mode="nearest",
            prefilter=False,
        ).astype(np.float64, copy=False)

    def classify_xyz(
        self,
        x: NDArray[np.float64],
        y: NDArray[np.float64],
        z: NDArray[np.float64],
    ) -> NDArray[np.uint8]:
        rows, cols = self._grid_coordinates(x, y)
        terrain_z = self._sample(
            self.ground_surface,
            rows,
            cols,
        )
        local_slope = self._sample(
            self.slope_surface,
            rows,
            cols,
        )
        tolerance = (
            self.params.threshold
            + self.params.scalar * local_slope
        )
        ground = np.abs(z - terrain_z) <= tolerance

        classes = np.full(
            z.shape[0],
            NON_GROUND_CLASS,
            dtype=np.uint8,
        )
        classes[ground] = GROUND_CLASS
        return classes


@dataclass(frozen=True, slots=True)
class SMRFResult:
    model: SMRFModel
    ground_count: int
    non_ground_count: int
    elapsed_seconds: float
    empty_cell_count: int
    low_outlier_cell_count: int
    object_cell_count: int
    inpainted_cell_count: int

    @property
    def point_count(self) -> int:
        return self.ground_count + self.non_ground_count


def _emit(
    callback: ProgressCallback | None,
    percent: int,
    message: str,
) -> None:
    if callback is not None:
        callback(max(0, min(100, int(percent))), message)


def _scaled_chunk(
    raw: NDArray,
    scale: float,
    offset: float,
) -> NDArray[np.float64]:
    values = np.asarray(raw, dtype=np.float64)
    values *= float(scale)
    values += float(offset)
    return values


def _iter_scaled_xyz(
    cloud: CloudModel,
    chunk_size: int,
):
    total = cloud.point_count
    scales = cloud.las.header.scales
    offsets = cloud.las.header.offsets
    raw_x = cloud.las.X
    raw_y = cloud.las.Y
    raw_z = cloud.las.Z

    for start in range(0, total, chunk_size):
        stop = min(start + chunk_size, total)
        yield (
            start,
            stop,
            _scaled_chunk(
                raw_x[start:stop],
                scales[0],
                offsets[0],
            ),
            _scaled_chunk(
                raw_y[start:stop],
                scales[1],
                offsets[1],
            ),
            _scaled_chunk(
                raw_z[start:stop],
                scales[2],
                offsets[2],
            ),
        )


@lru_cache(maxsize=64)
def _disk(radius: int) -> NDArray[np.bool_]:
    yy, xx = np.ogrid[
        -radius : radius + 1,
        -radius : radius + 1,
    ]
    return (xx * xx + yy * yy) <= radius * radius


def _nearest_fill(
    surface: NDArray[np.float32],
    missing: NDArray[np.bool_],
) -> NDArray[np.float32]:
    if not np.any(missing):
        return surface.astype(np.float32, copy=True)
    if np.all(missing):
        raise ValueError("SMRF grid contains no valid terrain cells")

    indices = distance_transform_edt(
        missing,
        return_distances=False,
        return_indices=True,
    )
    nearest = surface[tuple(indices)]
    filled = surface.astype(np.float32, copy=True)
    filled[missing] = nearest[missing]
    return filled


def _inpaint_surface(
    surface: NDArray[np.float32],
    missing: NDArray[np.bool_],
    params: SMRFParams,
) -> NDArray[np.float32]:
    """Fill void/object cells while keeping measured terrain cells fixed.

    The nearest-neighbour fill provides a stable initial estimate. Repeated
    four-neighbour relaxation then approximates the spring/harmonic inpainting
    used by the reference SMRF implementation, producing a continuous
    provisional terrain surface across removed objects and data voids.
    """

    missing = missing | ~np.isfinite(surface)
    values = _nearest_fill(surface, missing)
    if not np.any(missing):
        return values

    kernel = np.array(
        [
            [0.0, 0.25, 0.0],
            [0.25, 0.0, 0.25],
            [0.0, 0.25, 0.0],
        ],
        dtype=np.float32,
    )

    for _ in range(params.inpaint_iterations):
        relaxed = convolve(
            values,
            kernel,
            mode="nearest",
        )
        old = values[missing].copy()
        values[missing] = relaxed[missing]
        delta = float(
            np.max(np.abs(values[missing] - old))
        )
        if delta <= params.inpaint_tolerance:
            break

    return values.astype(np.float32, copy=False)


def _progressive_object_mask(
    surface: NDArray[np.float32],
    *,
    cell: float,
    slope: float,
    window: float,
    progress: Callable[[int, int], None] | None = None,
) -> NDArray[np.bool_]:
    """Reference-style SMRF progressive morphological filtering.

    Window radii increase linearly by one raster cell. Each opening starts
    from the previously opened surface. Cells are accumulated as objects when
    the surface drop exceeds slope * physical_window_radius.
    """

    maximum = max(1, int(ceil(window / cell)))
    last_surface = surface.astype(np.float32, copy=True)
    object_mask = np.zeros(
        surface.shape,
        dtype=np.bool_,
    )

    for radius in range(1, maximum + 1):
        opened = grey_opening(
            last_surface,
            footprint=_disk(radius),
            mode="nearest",
        ).astype(np.float32, copy=False)

        elevation_limit = slope * (radius * cell)
        object_mask |= (
            last_surface - opened
        ) > elevation_limit
        last_surface = opened

        if progress is not None:
            progress(radius, maximum)

    return object_mask


def _build_minimum_surface(
    cloud: CloudModel,
    params: SMRFParams,
    min_x: float,
    min_y: float,
    rows: int,
    cols: int,
    progress: ProgressCallback | None,
) -> tuple[NDArray[np.float32], NDArray[np.bool_]]:
    minimum_surface = np.full(
        (rows, cols),
        np.inf,
        dtype=np.float32,
    )
    flat_surface = minimum_surface.ravel()
    total = cloud.point_count

    for _, stop, x, y, z in _iter_scaled_xyz(
        cloud,
        params.chunk_size,
    ):
        col = np.floor(
            (x - min_x) / params.cell
        ).astype(np.int64, copy=False)
        row = np.floor(
            (y - min_y) / params.cell
        ).astype(np.int64, copy=False)
        np.clip(col, 0, cols - 1, out=col)
        np.clip(row, 0, rows - 1, out=row)
        flat = row * cols + col
        np.minimum.at(
            flat_surface,
            flat,
            z.astype(np.float32, copy=False),
        )
        _emit(
            progress,
            int(25 * stop / total),
            (
                "SMRF minimum surface: "
                f"{stop:,}/{total:,}"
            ),
        )

    empty = ~np.isfinite(minimum_surface)
    minimum_surface[empty] = np.nan
    return minimum_surface, empty


def _surface_slope(
    ground_surface: NDArray[np.float32],
    cell: float,
) -> NDArray[np.float32]:
    gy, gx = np.gradient(
        ground_surface.astype(np.float64),
        cell,
        cell,
    )
    slope = np.sqrt(gx * gx + gy * gy)
    return slope.astype(np.float32, copy=False)


def run_smrf(
    cloud: CloudModel,
    params: SMRFParams | None = None,
    progress: ProgressCallback | None = None,
) -> SMRFResult:
    """Classify ground with the Pingel SMRF processing sequence.

    Existing LAS classification is never used. The implementation follows the
    key SMRF stages: minimum surface, low-outlier removal, linear progressive
    morphology, object-cell removal, terrain inpainting, slope-aware point
    classification.
    """

    params = params or SMRFParams()
    params.validate()
    if cloud.point_count == 0:
        raise ValueError("Cannot classify an empty cloud")

    started = perf_counter()
    header = cloud.las.header
    min_x = float(header.mins[0])
    min_y = float(header.mins[1])
    max_x = float(header.maxs[0])
    max_y = float(header.maxs[1])

    cols = max(
        1,
        int(ceil((max_x - min_x) / params.cell)) + 1,
    )
    rows = max(
        1,
        int(ceil((max_y - min_y) / params.cell)) + 1,
    )
    grid_cells = rows * cols
    if grid_cells > params.max_grid_cells:
        raise ValueError(
            "SMRF raster would contain "
            f"{grid_cells:,} cells ({rows} x {cols}). "
            "Increase the Cell parameter."
        )

    LOGGER.info(
        "SMRF_START points=%d cell=%s slope=%s window=%s "
        "threshold=%s scalar=%s grid=%dx%d algorithm=PINGEL_R1",
        cloud.point_count,
        params.cell,
        params.slope,
        params.window,
        params.threshold,
        params.scalar,
        rows,
        cols,
    )

    minimum_surface, empty_cells = _build_minimum_surface(
        cloud,
        params,
        min_x,
        min_y,
        rows,
        cols,
        progress,
    )
    _emit(progress, 26, "SMRF filling empty minimum-surface cells")
    minimum_filled = _inpaint_surface(
        minimum_surface,
        empty_cells,
        params,
    )

    _emit(progress, 28, "SMRF detecting low outliers")
    low_outlier_cells = _progressive_object_mask(
        -minimum_filled,
        cell=params.cell,
        slope=5.0,
        window=params.cell,
    )

    def morphology_progress(
        radius: int,
        maximum: int,
    ) -> None:
        _emit(
            progress,
            30 + int(25 * radius / maximum),
            (
                "SMRF progressive morphology: "
                f"{radius}/{maximum}"
            ),
        )

    object_cells = _progressive_object_mask(
        minimum_filled,
        cell=params.cell,
        slope=params.slope,
        window=params.window,
        progress=morphology_progress,
    )

    inpaint_cells = (
        empty_cells
        | low_outlier_cells
        | object_cells
    )
    provisional_surface = minimum_filled.copy()
    provisional_surface[inpaint_cells] = np.nan

    _emit(
        progress,
        57,
        (
            "SMRF inpainting terrain holes: "
            f"{int(np.count_nonzero(inpaint_cells)):,} cells"
        ),
    )
    ground_surface = _inpaint_surface(
        provisional_surface,
        inpaint_cells,
        params,
    )
    slope_surface = _surface_slope(
        ground_surface,
        params.cell,
    )
    _emit(progress, 64, "SMRF provisional terrain ready")

    model = SMRFModel(
        params=params,
        min_x=min_x,
        min_y=min_y,
        rows=rows,
        cols=cols,
        ground_surface=ground_surface,
        slope_surface=slope_surface,
        object_cell_mask=object_cells,
        inpaint_cell_mask=inpaint_cells,
    )

    ground_count = 0
    total = cloud.point_count
    for _, stop, x, y, z in _iter_scaled_xyz(
        cloud,
        params.chunk_size,
    ):
        classes = model.classify_xyz(x, y, z)
        ground_count += int(
            np.count_nonzero(
                classes == GROUND_CLASS
            )
        )
        _emit(
            progress,
            64 + int(36 * stop / total),
            f"SMRF classify: {stop:,}/{total:,}",
        )

    non_ground_count = total - ground_count
    elapsed = perf_counter() - started
    empty_count = int(np.count_nonzero(empty_cells))
    low_count = int(
        np.count_nonzero(low_outlier_cells)
    )
    object_count = int(np.count_nonzero(object_cells))
    inpainted_count = int(
        np.count_nonzero(inpaint_cells)
    )

    LOGGER.info(
        "SMRF_RASTER empty=%d low_outlier=%d object=%d inpainted=%d",
        empty_count,
        low_count,
        object_count,
        inpainted_count,
    )
    LOGGER.info(
        "SMRF_DONE ground=%d non_ground=%d elapsed=%.3fs",
        ground_count,
        non_ground_count,
        elapsed,
    )
    _emit(progress, 100, "SMRF classification complete")

    return SMRFResult(
        model=model,
        ground_count=ground_count,
        non_ground_count=non_ground_count,
        elapsed_seconds=elapsed,
        empty_cell_count=empty_count,
        low_outlier_cell_count=low_count,
        object_cell_count=object_count,
        inpainted_cell_count=inpainted_count,
    )
