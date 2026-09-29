from __future__ import annotations

import logging
from dataclasses import dataclass
from math import ceil
from typing import Callable

import numpy as np
from scipy.ndimage import convolve, distance_transform_edt, map_coordinates

from ..cloud.model import CloudModel


LOGGER = logging.getLogger("las_cafiisica.classifiers.l3_cloth")
ProgressCallback = Callable[[int, str], None]


def _emit(
    callback: ProgressCallback | None,
    percent: int,
    message: str,
) -> None:
    if callback is not None:
        callback(max(0, min(100, int(percent))), message)


@dataclass(frozen=True, slots=True)
class L3ClothParams:
    sample_target: int = 3_000_000
    fine_resolution: float = 0.35
    coarse_resolution: float = 0.90
    threshold: float = 0.24
    iterations: int = 75


@dataclass(slots=True)
class _GridSurface:
    min_x: float
    min_y: float
    resolution: float
    rows: int
    cols: int
    z: np.ndarray

    def at(
        self,
        x: np.ndarray,
        y: np.ndarray,
    ) -> np.ndarray:
        col = (x - self.min_x) / self.resolution
        row = (y - self.min_y) / self.resolution
        np.clip(col, 0.0, self.cols - 1.0, out=col)
        np.clip(row, 0.0, self.rows - 1.0, out=row)
        return map_coordinates(
            self.z,
            [row, col],
            order=1,
            mode="nearest",
            prefilter=False,
        ).astype(np.float64, copy=False)


@dataclass(slots=True)
class L3ClothEvidence:
    params: L3ClothParams
    fine: _GridSurface
    coarse: _GridSurface
    sampled_return_count: int

    @property
    def synthetic_fill_point_count(self) -> int:
        return 0

    def fine_distance(
        self,
        x: np.ndarray,
        y: np.ndarray,
        z: np.ndarray,
    ) -> np.ndarray:
        return np.abs(z - self.fine.at(x, y))

    def coarse_distance(
        self,
        x: np.ndarray,
        y: np.ndarray,
        z: np.ndarray,
    ) -> np.ndarray:
        return np.abs(z - self.coarse.at(x, y))

    def tin_cloth_delta_proxy(
        self,
        x: np.ndarray,
        y: np.ndarray,
    ) -> np.ndarray:
        return self.fine.at(x, y) - self.coarse.at(x, y)

    def confidence_xyz(
        self,
        x: np.ndarray,
        y: np.ndarray,
        z: np.ndarray,
    ) -> np.ndarray:
        fine_d = self.fine_distance(x, y, z)
        coarse_d = self.coarse_distance(x, y, z)
        delta = np.abs(self.tin_cloth_delta_proxy(x, y))

        fine_score = np.exp(
            -np.square(
                fine_d / max(self.params.threshold, 1e-6)
            )
        )
        coarse_score = np.exp(
            -np.square(
                coarse_d
                / max(self.params.threshold * 2.0, 1e-6)
            )
        )
        persistence = np.exp(
            -np.square(
                delta
                / max(self.params.threshold * 2.5, 1e-6)
            )
        )
        return np.clip(
            0.55 * fine_score
            + 0.25 * coarse_score
            + 0.20 * persistence,
            0.0,
            1.0,
        )


def _sample_indices(
    count: int,
    target: int,
) -> np.ndarray:
    if count <= 0:
        return np.empty(0, dtype=np.int64)
    stride = max(1, int(ceil(count / max(1, target))))
    return np.arange(0, count, stride, dtype=np.int64)


def _scaled_xyz(
    cloud: CloudModel,
    indices: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    points = cloud.las.points[indices]
    scales = cloud.las.header.scales
    offsets = cloud.las.header.offsets
    x = np.asarray(points.X, dtype=np.float64) * scales[0] + offsets[0]
    y = np.asarray(points.Y, dtype=np.float64) * scales[1] + offsets[1]
    z = np.asarray(points.Z, dtype=np.float64) * scales[2] + offsets[2]
    return x, y, z


def _nearest_fill(surface: np.ndarray) -> np.ndarray:
    missing = ~np.isfinite(surface)
    if not np.any(missing):
        return surface
    if np.all(missing):
        raise RuntimeError("L3 cloth surface has no valid cells")
    indices = distance_transform_edt(
        missing,
        return_distances=False,
        return_indices=True,
    )
    out = surface.copy()
    out[missing] = surface[tuple(indices)][missing]
    return out


def _lower_envelope(
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    *,
    min_x: float,
    min_y: float,
    max_x: float,
    max_y: float,
    resolution: float,
) -> np.ndarray:
    cols = max(
        1,
        int(np.floor((max_x - min_x) / resolution)) + 1,
    )
    rows = max(
        1,
        int(np.floor((max_y - min_y) / resolution)) + 1,
    )
    surface = np.full((rows, cols), np.inf, dtype=np.float32)
    ix = np.floor((x - min_x) / resolution).astype(np.int64)
    iy = np.floor((y - min_y) / resolution).astype(np.int64)
    np.clip(ix, 0, cols - 1, out=ix)
    np.clip(iy, 0, rows - 1, out=iy)
    flat = iy * cols + ix
    np.minimum.at(
        surface.ravel(),
        flat,
        z.astype(np.float32, copy=False),
    )
    surface[~np.isfinite(surface)] = np.nan
    return _nearest_fill(surface)


def _adaptive_rigidity(
    terrain: np.ndarray,
    resolution: float,
) -> np.ndarray:
    gy, gx = np.gradient(terrain.astype(np.float64), resolution)
    slope_deg = np.degrees(
        np.arctan(np.hypot(gx, gy))
    )
    # Steep terrain must be less rigid so the cloth follows terrace faces.
    rigidity = np.full(terrain.shape, 0.82, dtype=np.float32)
    rigidity[(slope_deg >= 10.0) & (slope_deg < 30.0)] = 0.66
    rigidity[slope_deg >= 30.0] = 0.48
    return rigidity


def _relax_cloth(
    envelope: np.ndarray,
    *,
    resolution: float,
    iterations: int,
) -> np.ndarray:
    kernel = np.array(
        [
            [0.0, 0.25, 0.0],
            [0.25, 0.0, 0.25],
            [0.0, 0.25, 0.0],
        ],
        dtype=np.float32,
    )
    z_min = float(np.min(envelope))
    z_max = float(np.max(envelope))
    cloth = np.full_like(
        envelope,
        z_min - max(1.0, resolution * 2.0),
    )
    gravity_step = max(
        0.04,
        (z_max - z_min + 2.0) / max(1, iterations),
    )
    rigidity = _adaptive_rigidity(
        envelope,
        resolution,
    )

    for _ in range(iterations):
        lifted = cloth + gravity_step
        smooth = convolve(
            lifted,
            kernel,
            mode="nearest",
        )
        candidate = (
            rigidity * lifted
            + (1.0 - rigidity) * smooth
        )
        cloth = np.minimum(candidate, envelope)
    return cloth.astype(np.float32, copy=False)


def _build_surface(
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    *,
    min_x: float,
    min_y: float,
    max_x: float,
    max_y: float,
    resolution: float,
    iterations: int,
) -> _GridSurface:
    envelope = _lower_envelope(
        x,
        y,
        z,
        min_x=min_x,
        min_y=min_y,
        max_x=max_x,
        max_y=max_y,
        resolution=resolution,
    )
    cloth = _relax_cloth(
        envelope,
        resolution=resolution,
        iterations=iterations,
    )
    return _GridSurface(
        min_x=min_x,
        min_y=min_y,
        resolution=resolution,
        rows=cloth.shape[0],
        cols=cloth.shape[1],
        z=cloth,
    )


def build_l3_cloth(
    cloud: CloudModel,
    params: L3ClothParams | None = None,
    progress: ProgressCallback | None = None,
) -> L3ClothEvidence:
    params = params or L3ClothParams()
    names = set(cloud.las.point_format.dimension_names)
    if not {
        "return_number",
        "number_of_returns",
    }.issubset(names):
        raise RuntimeError(
            "L3 cloth requires ReturnNumber and NumberOfReturns"
        )

    _emit(progress, 2, "L3 cloth: sampling last/only returns")
    indices = _sample_indices(
        cloud.point_count,
        params.sample_target,
    )
    points = cloud.las.points[indices]
    rn = np.asarray(points.return_number, dtype=np.int16)
    nr = np.asarray(points.number_of_returns, dtype=np.int16)
    keep = (nr > 0) & (rn > 0) & (rn == nr)
    if "withheld" in names:
        keep &= ~np.asarray(points.withheld, dtype=np.bool_)

    selected = indices[keep]
    if selected.size < 100:
        raise RuntimeError(
            "L3 cloth has too few measured last/only returns"
        )

    x, y, z = _scaled_xyz(cloud, selected)
    min_x = float(cloud.las.header.mins[0])
    min_y = float(cloud.las.header.mins[1])
    max_x = float(cloud.las.header.maxs[0])
    max_y = float(cloud.las.header.maxs[1])

    _emit(progress, 28, "L3 cloth: coarse adaptive cloth")
    coarse = _build_surface(
        x,
        y,
        z,
        min_x=min_x,
        min_y=min_y,
        max_x=max_x,
        max_y=max_y,
        resolution=params.coarse_resolution,
        iterations=max(35, params.iterations // 2),
    )

    _emit(progress, 58, "L3 cloth: fine adaptive cloth")
    fine = _build_surface(
        x,
        y,
        z,
        min_x=min_x,
        min_y=min_y,
        max_x=max_x,
        max_y=max_y,
        resolution=params.fine_resolution,
        iterations=params.iterations,
    )

    LOGGER.info(
        "L3_CLOTH returns=%d fine=%.3f coarse=%.3f threshold=%.3f",
        selected.size,
        params.fine_resolution,
        params.coarse_resolution,
        params.threshold,
    )
    _emit(progress, 100, "L3 cloth ready")
    return L3ClothEvidence(
        params=params,
        fine=fine,
        coarse=coarse,
        sampled_return_count=int(selected.size),
    )
