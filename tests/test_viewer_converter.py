from __future__ import annotations

import laspy
import numpy as np

from las_classifier.classifiers.smrf import (
    GROUND_CLASS,
    NON_GROUND_CLASS,
    SMRFParams,
    run_smrf,
)
from las_classifier.cloud.loader import load_cloud
from las_classifier.viewer.converter import (
    _write_classified_viewer_laz,
)


def _scene(path):
    axis = np.arange(0.0, 31.0, 1.0)
    xs, ys = np.meshgrid(axis, axis)
    ground = 100.0 + 0.02 * xs + 0.01 * ys
    raised = (
        (xs >= 12.0)
        & (xs <= 18.0)
        & (ys >= 12.0)
        & (ys <= 18.0)
    )
    z = ground.copy()
    z[raised] += 5.0

    header = laspy.LasHeader(
        point_format=3,
        version="1.2",
    )
    las = laspy.LasData(header)
    las.x = xs.ravel()
    las.y = ys.ravel()
    las.z = z.ravel()
    las.classification = np.zeros(
        xs.size,
        dtype=np.uint8,
    )
    las.write(path)


def test_classified_viewer_cloud_is_sampled_and_keeps_fill(tmp_path):
    source = tmp_path / "scene.las"
    _scene(source)
    cloud = load_cloud(source)
    result = run_smrf(
        cloud,
        SMRFParams(
            cell=1.0,
            slope=0.15,
            window=8.0,
            threshold=0.5,
            scalar=1.25,
            fill_spacing=0.5,
            chunk_size=100,
        ),
    )

    output = tmp_path / "viewer.laz"
    _write_classified_viewer_laz(
        source,
        output,
        result,
        max_ground_points=80,
        max_non_ground_points=30,
    )

    viewer = laspy.read(output)
    classes = np.asarray(viewer.classification)
    synthetic = np.asarray(viewer.synthetic)

    synthetic_count = int(np.count_nonzero(synthetic))
    measured_count = len(viewer.points) - synthetic_count

    assert synthetic_count == result.synthetic_fill_point_count
    assert measured_count <= 110
    assert np.count_nonzero(classes == GROUND_CLASS) > 0
    assert np.count_nonzero(classes == NON_GROUND_CLASS) > 0
    assert np.all(
        classes[synthetic.astype(bool)]
        == GROUND_CLASS
    )
