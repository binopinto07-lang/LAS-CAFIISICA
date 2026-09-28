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
        np.arange(31, dtype=np.float64),
        np.arange(31, dtype=np.float64),
    )
    terrain = (
        100.0
        + 0.02 * xs
        + 0.01 * ys
    )
    object_mask = (
        (xs >= 12)
        & (xs <= 18)
        & (ys >= 12)
        & (ys <= 18)
    )
    z = terrain.copy()
    z[object_mask] += 5.0

    keep = np.ones(xs.shape, dtype=bool)
    keep[3:6, 23:26] = False

    header = laspy.LasHeader(
        point_format=3,
        version="1.2",
    )
    las = laspy.LasData(header)
    las.x = xs[keep]
    las.y = ys[keep]
    las.z = z[keep]

    # Deliberately misleading input: every point says "ground".
    las.classification = np.full(
        int(np.count_nonzero(keep)),
        2,
        dtype=np.uint8,
    )
    las.write(path)

    return (
        object_mask[keep].ravel(),
        terrain,
    )


def test_smrf_removes_object_and_inpaints_terrain(tmp_path):
    source = tmp_path / "scene.las"
    object_mask, terrain = _synthetic_scene(source)
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
            inpaint_iterations=400,
        ),
    )

    x = np.asarray(cloud.las.x)
    y = np.asarray(cloud.las.y)
    z = np.asarray(cloud.las.z)
    classes = result.model.classify_xyz(x, y, z)

    assert np.all(
        classes[object_mask]
        == NON_GROUND_CLASS
    )
    assert np.all(
        classes[~object_mask]
        == GROUND_CLASS
    )

    # The elevated 7x7 object must be removed from the terrain raster and
    # the provisional ground surface must be filled through that hole.
    assert result.object_cell_count > 0
    assert result.inpainted_cell_count > 0
    center = result.model.ground_surface[15, 15]
    assert abs(
        float(center) - float(terrain[15, 15])
    ) < 0.75
    assert np.all(
        np.isfinite(
            result.model.ground_surface
        )
    )

    np.testing.assert_array_equal(
        cloud.original_class,
        np.full(
            cloud.point_count,
            2,
            dtype=np.uint8,
        ),
    )


def test_smrf_uses_slope_scaled_final_threshold(tmp_path):
    path = tmp_path / "slope.las"
    xs, ys = np.meshgrid(
        np.arange(12, dtype=np.float64),
        np.arange(12, dtype=np.float64),
    )
    z = 50.0 + 0.25 * xs

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

    cloud = load_cloud(path)
    result = run_smrf(
        cloud,
        SMRFParams(
            cell=1.0,
            slope=0.30,
            window=6.0,
            threshold=0.20,
            scalar=1.25,
            chunk_size=64,
        ),
    )

    classes = result.model.classify_xyz(
        np.asarray(cloud.las.x),
        np.asarray(cloud.las.y),
        np.asarray(cloud.las.z),
    )
    assert np.all(classes == GROUND_CLASS)


def test_smrf_export_writes_classes_and_epsg_3763(tmp_path):
    source = tmp_path / "scene.las"
    object_mask, _ = _synthetic_scene(source)
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

    output = tmp_path / "classified.las"
    export_classified(
        source,
        output,
        result.model,
    )

    classified = laspy.read(output)
    crs = classified.header.parse_crs()
    output_classes = np.asarray(
        classified.classification
    )

    assert crs is not None
    assert crs.to_epsg() == 3763
    assert np.all(
        output_classes[object_mask]
        == NON_GROUND_CLASS
    )
    assert np.all(
        output_classes[~object_mask]
        == GROUND_CLASS
    )

    original = laspy.read(source)
    assert np.all(
        np.asarray(original.classification)
        == 2
    )
