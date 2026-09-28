from __future__ import annotations

import laspy
import numpy as np
from pyproj import CRS

from las_classifier.classifiers.adaptive_ptd import run_adaptive_ptd
from las_classifier.cloud.loader import load_cloud
from las_classifier.ground.ground_export import export_ground_only
from las_classifier.ground.types import GroundEngineParams


def test_ground_only_export_contains_only_class_2_and_preserves_crs(tmp_path):
    source = tmp_path / "input.las"
    axis = np.arange(0.0, 12.01, 0.50)
    xg, yg = np.meshgrid(axis, axis)
    zg = 50.0 + 0.20 * xg

    ground = np.column_stack((xg.ravel(), yg.ravel(), zg.ravel()))
    object_points = ground[::25].copy()
    object_points[:, 2] += 3.0
    xyz = np.vstack((ground, object_points))

    header = laspy.LasHeader(point_format=3, version="1.2")
    header.add_crs(CRS.from_epsg(3763))
    las = laspy.LasData(header)
    las.x = xyz[:, 0]
    las.y = xyz[:, 1]
    las.z = xyz[:, 2]
    las.classification = np.zeros(xyz.shape[0], dtype=np.uint8)
    las.write(source)

    cloud = load_cloud(source)
    result = run_adaptive_ptd(
        cloud,
        GroundEngineParams(
            sample_target=100_000,
            seed_resolution=4.0,
            candidate_spacing=0.50,
            max_iteration_distance=0.20,
            max_triangle_edge=10.0,
            synthetic_spacing=0.25,
            chunk_size=5000,
        ),
    )

    output = tmp_path / "output_ground.las"
    export_ground_only(
        source,
        output,
        result.model,
        include_synthetic=True,
    )

    exported = laspy.read(output)
    classes = np.asarray(exported.classification)
    assert classes.size > 0
    assert np.all(classes == 2)
    assert exported.header.parse_crs() is not None
    assert exported.header.parse_crs().to_epsg() == 3763
    assert len(exported.points) <= (
        len(las.points) + result.synthetic_fill_point_count
    )
