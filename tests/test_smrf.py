from __future__ import annotations

import laspy
import numpy as np

from las_classifier.classifiers.smrf import (
    GROUND_CLASS,
    NON_GROUND_CLASS,
    SMRFParams,
    run_smrf,
)
from las_classifier.cloud.exporter import export_classified
from las_classifier.cloud.loader import load_cloud


def _synthetic_scene(path):
    xs, ys = np.meshgrid(
        np.arange(25, dtype=np.float64),
        np.arange(25, dtype=np.float64),
    )
    ground = 100.0 + 0.01 * xs + 0.005 * ys
    object_mask = (
        (xs >= 10)
        & (xs <= 14)
        & (ys >= 10)
        & (ys <= 14)
    )
    z = ground.copy()
    z[object_mask] += 5.0

    header = laspy.LasHeader(
        point_format=3,
        version="1.2",
    )
    las = laspy.LasData(header)
    las.x = xs.ravel()
    las.y = ys.ravel()
    las.z = z.ravel()

    # Deliberately misleading source classification: every point is ground.
    las.classification = np.full(
        xs.size,
        2,
        dtype=np.uint8,
    )
    las.write(path)
    return object_mask.ravel()


def test_smrf_ignores_source_classification_and_detects_object(tmp_path):
    source = tmp_path / "scene.las"
    object_mask = _synthetic_scene(source)
    cloud = load_cloud(source)

    result = run_smrf(
        cloud,
        SMRFParams(
            cell=1.0,
            slope=0.15,
            window=8.0,
            threshold=0.5,
            scalar=1.25,
            chunk_size=100,
        ),
    )

    classes = result.model.classify_xyz(
        np.asarray(cloud.las.x),
        np.asarray(cloud.las.y),
        np.asarray(cloud.las.z),
    )

    assert np.all(
        classes[object_mask] == NON_GROUND_CLASS
    )
    assert np.all(
        classes[~object_mask] == GROUND_CLASS
    )
    assert result.non_ground_count == int(
        np.count_nonzero(object_mask)
    )
    np.testing.assert_array_equal(
        cloud.original_class,
        np.full(cloud.point_count, 2, dtype=np.uint8),
    )


def test_smrf_export_writes_classes_and_epsg_3763(tmp_path):
    source = tmp_path / "scene.las"
    object_mask = _synthetic_scene(source)
    cloud = load_cloud(source)
    result = run_smrf(
        cloud,
        SMRFParams(
            cell=1.0,
            window=8.0,
            threshold=0.5,
            chunk_size=100,
        ),
    )

    output = tmp_path / "classified.las"
    export_classified(source, output, result.model)

    classified = laspy.read(output)
    crs = classified.header.parse_crs()

    assert crs is not None
    assert crs.to_epsg() == 3763
    assert np.all(
        np.asarray(classified.classification)[object_mask]
        == NON_GROUND_CLASS
    )
    assert np.all(
        np.asarray(classified.classification)[~object_mask]
        == GROUND_CLASS
    )

    original = laspy.read(source)
    assert np.all(np.asarray(original.classification) == 2)
