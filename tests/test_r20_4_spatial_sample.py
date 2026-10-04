"""R20.4 terrain sampling must not depend on LAS point order."""
from __future__ import annotations

from pathlib import Path

import laspy
import numpy as np

from las_classifier.cloud.model import CloudModel
from las_classifier.ground.density import spatial_low_sample


def _cloud(order):
    header = laspy.LasHeader(point_format=3, version="1.2")
    header.scales = np.array([0.001, 0.001, 0.001])
    las = laspy.LasData(header)
    x = np.array([0.0, 0.1, 0.8, 0.9, 1.6, 1.7, 2.4, 2.5])
    y = np.array([0.0, 0.1, 0.0, 0.1, 0.0, 0.1, 0.0, 0.1])
    z = np.array([10.0, 12.0, 10.2, 13.0, 10.4, 11.5, 10.6, 14.0])
    las.x = x[order]
    las.y = y[order]
    las.z = z[order]
    las.classification = np.zeros(len(order), dtype=np.uint8)
    las.update_header()
    return CloudModel(
        path=Path("synthetic.las"),
        las=las,
        original_class=np.zeros(len(order), dtype=np.uint8),
    )


def test_spatial_low_sample_is_point_order_invariant():
    a = spatial_low_sample(_cloud(np.arange(8)), target=10_000)
    b = spatial_low_sample(_cloud(np.array([7, 2, 5, 0, 6, 1, 4, 3])), target=10_000)
    a = a[np.lexsort((a[:, 2], a[:, 1], a[:, 0]))]
    b = b[np.lexsort((b[:, 2], b[:, 1], b[:, 0]))]
    np.testing.assert_allclose(a, b)
