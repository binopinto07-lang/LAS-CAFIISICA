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
    _write_original_viewer_laz,
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


def test_original_viewer_is_normalized_for_potree_bounds(tmp_path):
    source = tmp_path / "precision_source.las"
    header = laspy.LasHeader(
        point_format=3,
        version="1.2",
    )
    header.scales = np.array(
        [1e-8, 1e-8, 1e-8],
        dtype=np.float64,
    )
    header.offsets = np.array(
        [55490.0, 162360.0, 177.0],
        dtype=np.float64,
    )
    las = laspy.LasData(header)
    count = 101
    step = np.arange(count, dtype=np.float64)
    las.x = 55490.275578279325 + step * 0.00012345
    las.y = 162360.3428948692 + step * 0.00023456
    las.z = 177.00068884629462 + step * 0.00034567
    las.classification = np.where(
        (step.astype(np.int64) % 2) == 0,
        2,
        1,
    ).astype(np.uint8)
    las.red = np.full(count, 12000, dtype=np.uint16)
    las.green = np.full(count, 22000, dtype=np.uint16)
    las.blue = np.full(count, 32000, dtype=np.uint16)
    las.write(source)

    output = tmp_path / "original_viewer.laz"
    _write_original_viewer_laz(
        source,
        output,
        max_points=20,
    )

    viewer = laspy.read(output)

    assert len(viewer.points) <= 20
    assert viewer.header.point_format.id == 3
    assert np.allclose(
        viewer.header.scales,
        [0.001, 0.001, 0.001],
    )
    assert np.all(np.isfinite(viewer.x))
    assert np.all(np.isfinite(viewer.y))
    assert np.all(np.isfinite(viewer.z))
    assert np.all(np.asarray(viewer.red) > 0)
    assert np.all(np.asarray(viewer.green) > 0)
    assert np.all(np.asarray(viewer.blue) > 0)

    mins = np.asarray(viewer.header.mins)
    maxs = np.asarray(viewer.header.maxs)
    xyz = np.column_stack(
        (
            np.asarray(viewer.x),
            np.asarray(viewer.y),
            np.asarray(viewer.z),
        )
    )
    assert np.all(xyz >= mins - 0.001)
    assert np.all(xyz <= maxs + 0.001)
