from __future__ import annotations

import logging
from math import ceil, sqrt

import numpy as np
from scipy.spatial import cKDTree

from ..cloud.model import CloudModel
from .types import GroundAnalysis


LOGGER = logging.getLogger("las_cafiisica.ground.density")


def deterministic_sample_indices(
    point_count: int,
    target: int,
) -> tuple[np.ndarray, int]:
    if point_count <= 0:
        return np.empty(0, dtype=np.int64), 1
    stride = max(1, int(ceil(point_count / max(1, target))))
    return np.arange(0, point_count, stride, dtype=np.int64), stride


def scaled_xyz(
    cloud: CloudModel,
    indices: np.ndarray,
) -> np.ndarray:
    scales = cloud.las.header.scales
    offsets = cloud.las.header.offsets
    x = np.asarray(cloud.las.X[indices], dtype=np.float64) * scales[0] + offsets[0]
    y = np.asarray(cloud.las.Y[indices], dtype=np.float64) * scales[1] + offsets[1]
    z = np.asarray(cloud.las.Z[indices], dtype=np.float64) * scales[2] + offsets[2]
    return np.column_stack((x, y, z))


def estimate_sample_spacing(
    xy: np.ndarray,
    max_probe: int = 160_000,
) -> float:
    if xy.shape[0] < 2:
        return 1.0
    if xy.shape[0] > max_probe:
        stride = int(ceil(xy.shape[0] / max_probe))
        probe = xy[::stride]
    else:
        probe = xy
    tree = cKDTree(probe)
    distances, _ = tree.query(probe, k=2, workers=-1)
    nn = distances[:, 1]
    nn = nn[np.isfinite(nn) & (nn > 0)]
    return float(np.median(nn)) if nn.size else 1.0


def analyze_cloud(
    cloud: CloudModel,
    sample_target: int = 2_500_000,
) -> tuple[GroundAnalysis, np.ndarray]:
    indices, stride = deterministic_sample_indices(
        cloud.point_count,
        sample_target,
    )
    xyz = scaled_xyz(cloud, indices)

    mins = cloud.las.header.mins
    maxs = cloud.las.header.maxs
    area = max(
        float((maxs[0] - mins[0]) * (maxs[1] - mins[1])),
        1e-9,
    )
    density = float(cloud.point_count / area)
    density_spacing = sqrt(1.0 / density) if density > 0 else 1.0

    sampled_spacing = estimate_sample_spacing(xyz[:, :2])
    # Deterministic thinning inflates nearest-neighbour spacing roughly with
    # sqrt(stride). Undo that inflation before using spacing for PTD thresholds.
    corrected_sample_spacing = sampled_spacing / sqrt(max(1, stride))
    spacing = max(
        0.005,
        min(
            corrected_sample_spacing,
            density_spacing * 2.0,
        ),
    )

    analysis = GroundAnalysis(
        point_count=cloud.point_count,
        median_spacing=float(spacing),
        xy_density=density,
        z_range=float(maxs[2] - mins[2]),
        sample_stride=stride,
        sample_count=int(xyz.shape[0]),
    )
    LOGGER.info(
        "GROUND_ANALYSIS points=%d sample=%d stride=%d spacing=%.4f "
        "sample_spacing=%.4f density_spacing=%.4f density=%.3f zrange=%.3f",
        analysis.point_count,
        analysis.sample_count,
        analysis.sample_stride,
        analysis.median_spacing,
        sampled_spacing,
        density_spacing,
        analysis.xy_density,
        analysis.z_range,
    )
    return analysis, xyz
