from __future__ import annotations

import laspy
import numpy as np

from las_classifier.cloud.loader import (
    IGNORE_INPUT_CLASSIFICATION,
    load_cloud,
)


def _write_cloud(path):
    header = laspy.LasHeader(
        point_format=3,
        version="1.2",
    )
    las = laspy.LasData(header)
    las.x = np.array([0.0, 1.0, 2.0])
    las.y = np.array([0.0, 1.0, 2.0])
    las.z = np.array([10.0, 11.0, 12.0])
    las.classification = np.array(
        [2, 5, 1],
        dtype=np.uint8,
    )
    las.write(path)


def test_loader_preserves_original_class_and_is_lazy(tmp_path):
    path = tmp_path / "sample.las"
    _write_cloud(path)

    cloud = load_cloud(path)

    assert IGNORE_INPUT_CLASSIFICATION is True
    np.testing.assert_array_equal(
        cloud.original_class,
        [2, 5, 1],
    )
    assert cloud._working_class is None
    assert cloud._xyz_cache is None

    np.testing.assert_array_equal(
        cloud.working_class,
        [0, 0, 0],
    )
    np.testing.assert_allclose(
        cloud.xyz[:, 2],
        [10.0, 11.0, 12.0],
    )
    assert cloud.point_count == 3


def test_loader_rejects_non_las_files(tmp_path):
    path = tmp_path / "sample.txt"
    path.write_text(
        "not a cloud",
        encoding="utf-8",
    )

    try:
        load_cloud(path)
    except ValueError as exc:
        assert "Unsupported point-cloud format" in str(exc)
    else:
        raise AssertionError("Expected ValueError")
