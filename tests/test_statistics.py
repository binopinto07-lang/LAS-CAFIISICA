from __future__ import annotations

import laspy
import numpy as np

from las_classifier.cloud.crs import WORKING_CRS, WORKING_EPSG
from las_classifier.cloud.loader import load_cloud
from las_classifier.cloud.statistics import calculate_statistics


def test_statistics_are_calculated_without_materializing_xyz(tmp_path):
    path = tmp_path / "stats.las"
    header = laspy.LasHeader(
        point_format=3,
        version="1.2",
    )
    las = laspy.LasData(header)
    las.x = np.array([0.0, 10.0, 0.0, 10.0])
    las.y = np.array([0.0, 0.0, 10.0, 10.0])
    las.z = np.array([100.0, 101.0, 105.0, 110.0])
    las.classification = np.array(
        [2, 2, 5, 5],
        dtype=np.uint8,
    )
    las.write(path)

    cloud = load_cloud(path)
    stats = calculate_statistics(cloud)

    assert cloud._xyz_cache is None
    assert cloud._working_class is None
    assert stats.point_count == 4
    assert stats.min_xyz == (0.0, 0.0, 100.0)
    assert stats.max_xyz == (10.0, 10.0, 110.0)
    assert stats.height_range == 10.0
    assert stats.crs == "EPSG:3763"
    assert stats.original_class_histogram == {
        2: 2,
        5: 2,
    }
    assert stats.approximate_xy_density == 0.04
    assert stats.approximate_point_spacing == 5.0
    assert "classification" in stats.dimensions


def test_working_crs_is_always_epsg_3763():
    assert WORKING_EPSG == 3763
    assert WORKING_CRS == "EPSG:3763"
