from __future__ import annotations

import laspy
import numpy as np

from las_classifier.cloud.loader import load_cloud


def test_laz_roundtrip_preserves_original_class(tmp_path):
    path = tmp_path / "sample.laz"
    header = laspy.LasHeader(point_format=3, version="1.2")
    las = laspy.LasData(header)
    las.x = np.array([0.0, 1.0])
    las.y = np.array([0.0, 1.0])
    las.z = np.array([5.0, 6.0])
    las.classification = np.array([2, 5], dtype=np.uint8)
    las.write(path)

    cloud = load_cloud(path)

    np.testing.assert_array_equal(cloud.original_class, [2, 5])
    np.testing.assert_array_equal(cloud.working_class, [0, 0])
