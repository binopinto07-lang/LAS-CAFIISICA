from __future__ import annotations

import logging
from dataclasses import dataclass
from math import ceil
from typing import Callable

import numpy as np

from ..cloud.model import CloudModel


LOGGER = logging.getLogger("las_cafiisica.terrain.l3_support")
ProgressCallback = Callable[[int, str], None]


@dataclass(frozen=True, slots=True)
class L3SupportParams:
    sample_target: int = 4_000_000
    voxel_size: float = 0.45
    max_plane_distance: float = 0.32
    min_ptd_score: float = 0.24
    saturation_count: int = 8


@dataclass(slots=True)
class L3SupportGrid:
    params: L3SupportParams
    origin: np.ndarray
    dimensions: tuple[int, int, int]
    keys: np.ndarray
    counts: np.ndarray
    mean_ptd: np.ndarray
    sampled_candidate_count: int

    def _keys_for(
        self,
        x: np.ndarray,
        y: np.ndarray,
        z: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        xyz = np.column_stack((x, y, z))
        ijk = np.floor(
            (xyz - self.origin[None, :])
            / self.params.voxel_size
        ).astype(np.int64)
        nx, ny, nz = self.dimensions
        inside = (
            (ijk[:, 0] >= 0)
            & (ijk[:, 1] >= 0)
            & (ijk[:, 2] >= 0)
            & (ijk[:, 0] < nx)
            & (ijk[:, 1] < ny)
            & (ijk[:, 2] < nz)
        )
        keys = np.full(x.shape[0], -1, dtype=np.int64)
        if np.any(inside):
            local = ijk[inside]
            keys[inside] = (
                local[:, 0]
                + nx * (
                    local[:, 1]
                    + ny * local[:, 2]
                )
            )
        return keys, inside

    def score_xyz(
        self,
        x: np.ndarray,
        y: np.ndarray,
        z: np.ndarray,
    ) -> np.ndarray:
        score = np.zeros(x.shape[0], dtype=np.float64)
        if self.keys.size == 0 or x.size == 0:
            return score

        query, inside = self._keys_for(x, y, z)
        ids = np.flatnonzero(inside)
        if ids.size == 0:
            return score

        pos = np.searchsorted(self.keys, query[ids])
        valid = pos < self.keys.size
        safe = np.minimum(pos, max(0, self.keys.size - 1))
        valid &= self.keys[safe] == query[ids]
        if not np.any(valid):
            return score

        hit_ids = ids[valid]
        hit_pos = pos[valid]
        count_score = np.clip(
            np.log1p(self.counts[hit_pos])
            / np.log1p(max(1, self.params.saturation_count)),
            0.0,
            1.0,
        )
        score[hit_ids] = np.clip(
            0.68 * count_score
            + 0.32 * self.mean_ptd[hit_pos],
            0.0,
            1.0,
        )
        return score


def _sample_indices(
    count: int,
    target: int,
) -> np.ndarray:
    if count <= 0:
        return np.empty(0, dtype=np.int64)
    stride = max(
        1,
        int(ceil(count / max(1, target))),
    )
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


def build_l3_support_grid(
    cloud: CloudModel,
    ptd_model,
    params: L3SupportParams | None = None,
    progress: ProgressCallback | None = None,
) -> L3SupportGrid:
    params = params or L3SupportParams()
    names = set(cloud.las.point_format.dimension_names)
    if not {
        "return_number",
        "number_of_returns",
    }.issubset(names):
        raise RuntimeError(
            "L3 support grid requires return metadata"
        )

    if progress is not None:
        progress(5, "L3 support: sampling measured returns")

    indices = _sample_indices(
        cloud.point_count,
        params.sample_target,
    )
    points = cloud.las.points[indices]
    rn = np.asarray(points.return_number, dtype=np.int16)
    nr = np.asarray(points.number_of_returns, dtype=np.int16)
    last_or_only = (
        (nr > 0)
        & (rn > 0)
        & (rn == nr)
    )
    if "withheld" in names:
        last_or_only &= ~np.asarray(
            points.withheld,
            dtype=np.bool_,
        )

    x, y, z = _scaled_xyz(cloud, indices)
    ptd_score, metrics = ptd_model._confidence(
        x,
        y,
        z,
        points=points,
    )
    candidate = (
        last_or_only
        & metrics["valid"]
        & np.isfinite(metrics["plane_distance"])
        & (
            metrics["plane_distance"]
            <= params.max_plane_distance
        )
        & (ptd_score >= params.min_ptd_score)
    )
    candidate_ids = np.flatnonzero(candidate)

    mins = np.asarray(cloud.las.header.mins, dtype=np.float64)
    maxs = np.asarray(cloud.las.header.maxs, dtype=np.float64)
    origin = (
        np.floor(mins / params.voxel_size)
        * params.voxel_size
    )
    dims = np.maximum(
        1,
        np.ceil(
            (maxs - origin) / params.voxel_size
        ).astype(np.int64)
        + 2,
    )
    nx, ny, nz = map(int, dims)

    if candidate_ids.size == 0:
        return L3SupportGrid(
            params=params,
            origin=origin,
            dimensions=(nx, ny, nz),
            keys=np.empty(0, dtype=np.int64),
            counts=np.empty(0, dtype=np.int32),
            mean_ptd=np.empty(0, dtype=np.float32),
            sampled_candidate_count=0,
        )

    xyz = np.column_stack(
        (
            x[candidate_ids],
            y[candidate_ids],
            z[candidate_ids],
        )
    )
    ijk = np.floor(
        (xyz - origin[None, :])
        / params.voxel_size
    ).astype(np.int64)
    keys = (
        ijk[:, 0]
        + nx * (
            ijk[:, 1]
            + ny * ijk[:, 2]
        )
    )

    unique_keys, inverse = np.unique(
        keys,
        return_inverse=True,
    )
    counts = np.bincount(
        inverse,
        minlength=unique_keys.size,
    ).astype(np.int32)
    score_sum = np.bincount(
        inverse,
        weights=ptd_score[candidate_ids],
        minlength=unique_keys.size,
    )
    mean_ptd = (
        score_sum / np.maximum(counts, 1)
    ).astype(np.float32)

    LOGGER.info(
        "L3_SUPPORT sampled=%d candidates=%d voxels=%d",
        indices.size,
        candidate_ids.size,
        unique_keys.size,
    )
    if progress is not None:
        progress(
            100,
            f"L3 support: {unique_keys.size:,} occupied voxels",
        )

    return L3SupportGrid(
        params=params,
        origin=origin,
        dimensions=(nx, ny, nz),
        keys=unique_keys.astype(np.int64, copy=False),
        counts=counts,
        mean_ptd=mean_ptd,
        sampled_candidate_count=int(candidate_ids.size),
    )
