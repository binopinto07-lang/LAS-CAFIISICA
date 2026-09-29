from __future__ import annotations

import laspy
import numpy as np

from las_classifier.classifiers.l3_cloth import (
    L3ClothParams,
    build_l3_cloth,
)
from las_classifier.cloud.loader import load_cloud


def test_l3_cloth_uses_measured_last_only_returns(tmp_path):
    path = tmp_path / "l3_cloth.las"
    header = laspy.LasHeader(
        point_format=3,
        version="1.2",
    )
    las = laspy.LasData(header)

    cols = 20
    rows = 12
    xx, yy = np.meshgrid(
        np.arange(cols, dtype=np.float64) * 0.25,
        np.arange(rows, dtype=np.float64) * 0.25,
    )
    x = xx.ravel()
    y = yy.ravel()
    z = 100.0 + 0.15 * x + 0.05 * y
    count = x.size

    las.x = x
    las.y = y
    las.z = z
    las.classification = np.ones(count, dtype=np.uint8)
    las.return_number = np.full(count, 2, dtype=np.uint8)
    las.number_of_returns = np.full(count, 2, dtype=np.uint8)

    # A small subset is first-return-only noise and must not drive the cloth.
    las.return_number[:30] = 1
    las.number_of_returns[:30] = 2
    las.z[:30] = z[:30] + 4.0
    las.write(path)

    cloud = load_cloud(path)
    cloth = build_l3_cloth(
        cloud,
        L3ClothParams(
            sample_target=10_000,
            fine_resolution=0.50,
            coarse_resolution=1.00,
            threshold=0.30,
            iterations=20,
        ),
    )

    assert cloth.sampled_return_count == count - 30

    query_x = np.array([2.0])
    query_y = np.array([1.0])
    query_z = np.array([100.0 + 0.15 * 2.0 + 0.05 * 1.0])
    confidence = cloth.confidence_xyz(
        query_x,
        query_y,
        query_z,
    )

    assert np.isfinite(cloth.fine.z).all()
    assert np.isfinite(cloth.coarse.z).all()
    assert 0.0 <= float(confidence[0]) <= 1.0
