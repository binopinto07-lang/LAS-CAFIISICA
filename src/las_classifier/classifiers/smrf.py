from __future__ import annotations

import logging
from dataclasses import dataclass
from math import ceil
from time import perf_counter
from typing import Callable

import numpy as np
from numpy.typing import NDArray
from scipy.ndimage import distance_transform_edt, grey_opening

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

    def validate(self) -> None:
        if self.cell <= 0:
            raise ValueError("SMRF cell must be greater than zero")
        if self.slope < 0:
            raise ValueError("SMRF slope cannot be negative")
        if self.window <= 0:
            raise ValueError("SMRF window must be greater than zero")
        if self.threshold < 0:
            raise ValueError("SMRF threshold cannot be negative")
        if self.scalar <= 0:
            raise ValueError("SMRF scalar must be greater than zero")
        if self.chunk_size < 1:
            raise ValueError("SMRF chunk_size must be positive")
        if self.max_grid_cells < 1:
            raise ValueError("SMRF max_grid_cells must be positive")


@dataclass(frozen=True, slots=True)
class SMRFModel:
    params: SMRFParams
    min_x: float
    min_y: float
    rows: int
    cols: int
    ground_surface: NDArray[np.float32]

    def _indices(
        self,
        x: NDArray[np.float64],
        y: NDArray[np.float64],
    ) -> tuple[NDArray[np.int64], NDArray[np.int64]]:
        cols = np.floor((x - self.min_x) / self.params.cell).astype(
            np.int64,
            copy=False,
        )
        rows = np.floor((y - self.min_y) / self.params.cell).astype(
            np.int64,
            copy=False,
        )
        np.clip(cols, 0, self.cols - 1, out=cols)
        np.clip(rows, 0, self.rows - 1, out=rows)
        return rows, cols

    def classify_xyz(
        self,
        x: NDArray[np.float64],
        y: NDArray[np.float64],
        z: NDArray[np.float64],
    ) -> NDArray[np.uint8]:
        rows, cols = self._indices(x, y)
        terrain_z = self.ground_surface[rows, cols]
        ground = z <= (
            terrain_z.astype(np.float64, copy=False)
            + self.params.threshold
        )
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
            _scaled_chunk(raw_x[start:stop], scales[0], offsets[0]),
            _scaled_chunk(raw_y[start:stop], scales[1], offsets[1]),
            _scaled_chunk(raw_z[start:stop], scales[2], offsets[2]),
        )


def _window_radii(params: SMRFParams) -> list[int]:
    maximum = max(1, int(ceil(params.window / params.cell)))
    radii: list[int] = []
    radius = 1
    while radius < maximum:
        radii.append(radius)
        radius *= 2
    radii.append(maximum)
    return sorted(set(radii))


def _fill_empty_cells(surface: NDArray[np.float32]) -> NDArray[np.float32]:
    empty = ~np.isfinite(surface)
    if not np.any(empty):
        return surface
    if np.all(empty):
        raise ValueError("SMRF grid contains no points")

    nearest = distance_transform_edt(
        empty,
        return_distances=False,
        return_indices=True,
    )
    return surface[tuple(nearest)].astype(np.float32, copy=False)


def _progressive_surface(
    minimum_surface: NDArray[np.float32],
    params: SMRFParams,
    callback: ProgressCallback | None,
) -> NDArray[np.float32]:
    surface = _fill_empty_cells(minimum_surface)
    radii = _window_radii(params)
    largest_supported = max(
        1,
        (max(surface.shape) - 1) // 2,
    )

    for index, requested_radius in enumerate(radii, start=1):
        radius = min(requested_radius, largest_supported)
        size = 2 * radius + 1
        opened = grey_opening(
            surface,
            size=(size, size),
            mode="nearest",
        ).astype(np.float32, copy=False)

        distance = radius * params.cell
        limit = (
            params.threshold
            + params.scalar * params.slope * distance
        )
        delta = surface - opened
        surface = np.where(
            delta > limit,
            opened,
            surface,
        ).astype(np.float32, copy=False)

        _emit(
            callback,
            45 + int(20 * index / len(radii)),
            f"SMRF morphology {index}/{len(radii)}",
        )

    return surface


def run_smrf(
    cloud: CloudModel,
    params: SMRFParams | None = None,
    progress: ProgressCallback | None = None,
) -> SMRFResult:
    """Classify ground from geometry only.

    Existing LAS classification is never consulted. The algorithm creates a
    minimum-elevation raster, progressively removes elevated objects with
    morphological openings, then classifies points against the terrain model.
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

    cols = max(1, int(ceil((max_x - min_x) / params.cell)) + 1)
    rows = max(1, int(ceil((max_y - min_y) / params.cell)) + 1)
    grid_cells = rows * cols
    if grid_cells > params.max_grid_cells:
        raise ValueError(
            "SMRF raster would contain "
            f"{grid_cells:,} cells ({rows} x {cols}). "
            "Increase the Cell parameter."
        )

    LOGGER.info(
        "SMRF_START points=%d cell=%s slope=%s window=%s "
        "threshold=%s scalar=%s grid=%dx%d",
        cloud.point_count,
        params.cell,
        params.slope,
        params.window,
        params.threshold,
        params.scalar,
        rows,
        cols,
    )

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
        col = np.floor((x - min_x) / params.cell).astype(
            np.int64,
            copy=False,
        )
        row = np.floor((y - min_y) / params.cell).astype(
            np.int64,
            copy=False,
        )
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
            int(45 * stop / total),
            f"SMRF minimum surface: {stop:,}/{total:,}",
        )

    ground_surface = _progressive_surface(
        minimum_surface,
        params,
        progress,
    )
    model = SMRFModel(
        params=params,
        min_x=min_x,
        min_y=min_y,
        rows=rows,
        cols=cols,
        ground_surface=ground_surface,
    )

    ground_count = 0
    processed = 0
    for _, stop, x, y, z in _iter_scaled_xyz(
        cloud,
        params.chunk_size,
    ):
        classes = model.classify_xyz(x, y, z)
        ground_count += int(np.count_nonzero(classes == GROUND_CLASS))
        processed = stop
        _emit(
            progress,
            65 + int(35 * processed / total),
            f"SMRF classify: {processed:,}/{total:,}",
        )

    non_ground_count = total - ground_count
    elapsed = perf_counter() - started
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
    )
