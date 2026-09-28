from __future__ import annotations

import laspy
import numpy as np
from pyproj import CRS

from las_classifier.classifiers.adaptive_ptd import (
    GROUND_CLASS,
    NON_GROUND_CLASS,
    run_adaptive_ptd,
)
from las_classifier.classifiers.hybrid_ground import run_hybrid_ground
from las_classifier.cloud.loader import load_cloud
from las_classifier.ground.types import GroundEngineParams


def _terrain_with_tree_gap(path):
    axis = np.arange(0.0, 20.01, 0.25)
    xg, yg = np.meshgrid(axis, axis)
    zg = 100.0 + 0.60 * xg + 0.10 * yg

    gap = (
        (xg >= 9.0)
        & (xg <= 11.0)
        & (yg >= 9.0)
        & (yg <= 11.0)
    )
    ground = np.column_stack((xg[~gap], yg[~gap], zg[~gap]))

    rng = np.random.default_rng(42)
    vegetation_count = 1400
    vx = rng.uniform(9.0, 11.0, vegetation_count)
    vy = rng.uniform(9.0, 11.0, vegetation_count)
    terrain_z = 100.0 + 0.60 * vx + 0.10 * vy
    vz = terrain_z + rng.uniform(1.2, 4.5, vegetation_count)
    vegetation = np.column_stack((vx, vy, vz))

    xyz = np.vstack((ground, vegetation))
    header = laspy.LasHeader(point_format=3, version="1.2")
    header.add_crs(CRS.from_epsg(3763))
    las = laspy.LasData(header)
    las.x = xyz[:, 0]
    las.y = xyz[:, 1]
    las.z = xyz[:, 2]
    las.classification = np.full(xyz.shape[0], 2, dtype=np.uint8)
    las.write(path)
    return ground.shape[0], vegetation.shape[0]


def _params():
    return GroundEngineParams(
        quality="high",
        sample_target=200_000,
        seed_resolution=4.0,
        candidate_spacing=0.40,
        max_iteration_angle_deg=18.0,
        max_iteration_distance=0.18,
        max_iterations=8,
        confidence_threshold=0.62,
        max_triangle_edge=10.0,
        gap_max_size=6.0,
        synthetic_spacing=0.25,
        chunk_size=10_000,
    )


def test_adaptive_ptd_preserves_slope_rejects_tree_and_rebuilds_gap(tmp_path):
    source = tmp_path / "slope_tree.las"
    ground_n, _ = _terrain_with_tree_gap(source)
    cloud = load_cloud(source)

    result = run_adaptive_ptd(cloud, _params())
    classes = result.model.classify_xyz(
        np.asarray(cloud.las.x),
        np.asarray(cloud.las.y),
        np.asarray(cloud.las.z),
    )

    ground_classes = classes[:ground_n]
    vegetation_classes = classes[ground_n:]

    assert np.mean(ground_classes == GROUND_CLASS) > 0.97
    assert np.mean(vegetation_classes == NON_GROUND_CLASS) > 0.98
    assert result.supported_gap_count > 0
    assert result.synthetic_fill_point_count > 0


def test_hybrid_does_not_promote_obvious_high_object(tmp_path):
    source = tmp_path / "hybrid_scene.las"
    ground_n, _ = _terrain_with_tree_gap(source)
    cloud = load_cloud(source)

    result = run_hybrid_ground(cloud, _params())
    classes = result.model.classify_xyz(
        np.asarray(cloud.las.x),
        np.asarray(cloud.las.y),
        np.asarray(cloud.las.z),
    )

    assert np.mean(classes[:ground_n] == GROUND_CLASS) > 0.94
    assert np.mean(classes[ground_n:] == NON_GROUND_CLASS) > 0.98
    assert result.engine_name == "Hybrid"
