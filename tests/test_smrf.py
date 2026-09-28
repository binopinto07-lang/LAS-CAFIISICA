from __future__ import annotations

import laspy
import numpy as np

from las_classifier.classifiers.smrf import (
    GROUND_CLASS,
    NON_GROUND_CLASS,
    SMRFParams,
    run_smrf,
)
from las_classifier.cloud.exporter import (
    export_classified,
)
from las_classifier.cloud.loader import (
    load_cloud,
)


def _synthetic_scene(path):
    xs, ys = np.meshgrid(
        np.arange(
            31,
            dtype=np.float64,
        ),
        np.arange(
            31,
            dtype=np.float64,
        ),
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

    # A genuine enclosed data void is also rebuilt.
    keep = np.ones(
        xs.shape,
        dtype=bool,
    )
    keep[3:6, 23:26] = False

    header = laspy.LasHeader(
        point_format=3,
        version="1.2",
    )
    las = laspy.LasData(
        header
    )
    las.x = xs[keep]
    las.y = ys[keep]
    las.z = z[keep]

    # Deliberately wrong source labels: everything says ground.
    las.classification = np.full(
        int(
            np.count_nonzero(
                keep
            )
        ),
        2,
        dtype=np.uint8,
    )
    las.write(path)

    return (
        object_mask[keep].ravel(),
        terrain,
    )


def test_smrf_removes_object_and_builds_fill_surface(
    tmp_path,
):
    source = tmp_path / "scene.las"
    (
        object_mask,
        terrain,
    ) = _synthetic_scene(
        source
    )
    cloud = load_cloud(
        source
    )

    result = run_smrf(
        cloud,
        SMRFParams(
            cell=1.0,
            slope=0.15,
            window=8.0,
            threshold=0.5,
            scalar=1.25,
            fill_spacing=0.25,
            chunk_size=100,
            inpaint_iterations=400,
        ),
    )

    x = np.asarray(
        cloud.las.x
    )
    y = np.asarray(
        cloud.las.y
    )
    z = np.asarray(
        cloud.las.z
    )
    classes = (
        result.model.classify_xyz(
            x,
            y,
            z,
        )
    )

    assert np.all(
        classes[object_mask]
        == NON_GROUND_CLASS
    )
    assert np.all(
        classes[~object_mask]
        == GROUND_CLASS
    )

    assert (
        result.object_cell_count
        > 0
    )
    assert (
        result.interior_empty_cell_count
        >= 9
    )
    assert (
        result.synthetic_fill_point_count
        > result.inpainted_cell_count
    )

    center = (
        result.model.ground_surface[
            15,
            15,
        ]
    )
    assert abs(
        float(center)
        - float(
            terrain[15, 15]
        )
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


def test_smrf_rejects_tall_vegetation_on_steep_face(
    tmp_path,
):
    path = tmp_path / "steep.las"
    axis = np.arange(
        0.0,
        21.0,
        0.5,
    )
    xs, ys = np.meshgrid(
        axis,
        axis,
    )
    ground_z = (
        100.0
        + 1.20 * xs
        + 0.20 * ys
    )

    gx = xs.ravel()
    gy = ys.ravel()
    gz = ground_z.ravel()

    vegetation_sites = (
        (gx >= 8.0)
        & (gx <= 12.0)
        & (gy >= 8.0)
        & (gy <= 12.0)
    )
    vx = np.concatenate(
        [
            gx[vegetation_sites],
            gx[vegetation_sites],
        ]
    )
    vy = np.concatenate(
        [
            gy[vegetation_sites],
            gy[vegetation_sites],
        ]
    )
    vz = np.concatenate(
        [
            gz[vegetation_sites]
            + 3.0,
            gz[vegetation_sites]
            + 5.0,
        ]
    )

    header = laspy.LasHeader(
        point_format=3,
        version="1.2",
    )
    las = laspy.LasData(
        header
    )
    las.x = np.concatenate(
        [gx, vx]
    )
    las.y = np.concatenate(
        [gy, vy]
    )
    las.z = np.concatenate(
        [gz, vz]
    )
    las.classification = np.zeros(
        las.x.shape[0],
        dtype=np.uint8,
    )
    las.write(path)

    cloud = load_cloud(path)
    result = run_smrf(
        cloud,
        SMRFParams(
            cell=1.0,
            slope=0.15,
            window=8.0,
            threshold=0.5,
            scalar=1.25,
            fill_spacing=0.25,
            chunk_size=256,
        ),
    )
    classes = (
        result.model.classify_xyz(
            np.asarray(
                cloud.las.x
            ),
            np.asarray(
                cloud.las.y
            ),
            np.asarray(
                cloud.las.z
            ),
        )
    )

    ground_classes = classes[
        : gx.size
    ]
    vegetation_classes = classes[
        gx.size :
    ]

    assert (
        np.mean(
            ground_classes
            == GROUND_CLASS
        )
        > 0.98
    )
    assert np.all(
        vegetation_classes
        == NON_GROUND_CLASS
    )


def test_smrf_export_adds_synthetic_ground_points(
    tmp_path,
):
    source = tmp_path / "scene.las"
    (
        object_mask,
        _,
    ) = _synthetic_scene(
        source
    )
    cloud = load_cloud(
        source
    )
    result = run_smrf(
        cloud,
        SMRFParams(
            cell=1.0,
            slope=0.15,
            window=8.0,
            threshold=0.5,
            scalar=1.25,
            fill_spacing=0.25,
            chunk_size=100,
        ),
    )

    output = (
        tmp_path
        / "classified.las"
    )
    export_classified(
        source,
        output,
        result.model,
    )

    classified = laspy.read(
        output
    )
    crs = (
        classified.header.parse_crs()
    )
    output_classes = np.asarray(
        classified.classification
    )

    assert crs is not None
    assert (
        crs.to_epsg()
        == 3763
    )
    assert len(
        classified.points
    ) == (
        cloud.point_count
        + result.synthetic_fill_point_count
    )

    original_part = output_classes[
        : cloud.point_count
    ]
    assert np.all(
        original_part[
            object_mask
        ]
        == NON_GROUND_CLASS
    )
    assert np.all(
        output_classes[
            cloud.point_count :
        ]
        == GROUND_CLASS
    )

    synthetic = np.asarray(
        classified.synthetic
    )
    assert int(
        np.count_nonzero(
            synthetic
        )
    ) == (
        result.synthetic_fill_point_count
    )

    original = laspy.read(
        source
    )
    assert np.all(
        np.asarray(
            original.classification
        )
        == 2
    )
