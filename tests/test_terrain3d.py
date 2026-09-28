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


def _encode_normals(
    normals: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    values = np.clip(
        np.rint(normals * 127.0),
        -127,
        127,
    ).astype(np.int8)
    return (
        values[:, 0],
        values[:, 1],
        values[:, 2],
    )


def test_terrain3d_keeps_steep_face_and_rejects_vegetation(
    tmp_path,
):
    path = tmp_path / "steep_terrain3d.las"

    y = np.arange(
        0.0,
        10.0,
        0.20,
    )
    bottom_x = np.arange(
        0.0,
        10.0,
        0.20,
    )
    top_x = np.arange(
        11.5,
        21.5,
        0.20,
    )
    face_x = np.arange(
        10.0,
        11.6,
        0.05,
    )

    bx, by = np.meshgrid(
        bottom_x,
        y,
    )
    tx, ty = np.meshgrid(
        top_x,
        y,
    )
    fx, fy = np.meshgrid(
        face_x,
        y,
    )

    face_slope = 3.0
    bz = np.zeros_like(bx)
    fz = face_slope * (
        fx - 10.0
    )
    top_z = face_slope * 1.5
    tz = np.full_like(
        tx,
        top_z,
    )

    bottom = np.column_stack(
        (
            bx.ravel(),
            by.ravel(),
            bz.ravel(),
        )
    )
    face = np.column_stack(
        (
            fx.ravel(),
            fy.ravel(),
            fz.ravel(),
        )
    )
    top = np.column_stack(
        (
            tx.ravel(),
            ty.ravel(),
            tz.ravel(),
        )
    )

    bottom_n = np.tile(
        [0.0, 0.0, 1.0],
        (bottom.shape[0], 1),
    )
    face_normal = np.array(
        [-face_slope, 0.0, 1.0],
        dtype=np.float64,
    )
    face_normal /= np.linalg.norm(
        face_normal
    )
    face_n = np.tile(
        face_normal,
        (face.shape[0], 1),
    )
    top_n = np.tile(
        [0.0, 0.0, 1.0],
        (top.shape[0], 1),
    )

    rng = np.random.default_rng(7)
    vegetation_count = 6000
    vx = rng.normal(
        10.8,
        0.55,
        vegetation_count,
    )
    vy = rng.normal(
        5.0,
        0.80,
        vegetation_count,
    )
    local_ground = face_slope * np.clip(
        vx - 10.0,
        0.0,
        1.5,
    )
    vz = (
        local_ground
        + rng.uniform(
            0.8,
            4.5,
            vegetation_count,
        )
    )
    vegetation = np.column_stack(
        (vx, vy, vz)
    )
    vegetation_n = rng.normal(
        size=(
            vegetation_count,
            3,
        )
    )
    vegetation_n /= np.linalg.norm(
        vegetation_n,
        axis=1,
    )[:, None]

    terrain = np.vstack(
        (bottom, face, top)
    )
    terrain_normals = np.vstack(
        (bottom_n, face_n, top_n)
    )
    xyz = np.vstack(
        (terrain, vegetation)
    )
    normals = np.vstack(
        (
            terrain_normals,
            vegetation_n,
        )
    )

    header = laspy.LasHeader(
        point_format=3,
        version="1.2",
    )
    las = laspy.LasData(header)
    las.add_extra_dims(
        [
            laspy.ExtraBytesParams(
                name="normal x",
                type=np.int8,
            ),
            laspy.ExtraBytesParams(
                name="normal y",
                type=np.int8,
            ),
            laspy.ExtraBytesParams(
                name="normal z",
                type=np.int8,
            ),
        ]
    )
    las.x = xyz[:, 0]
    las.y = xyz[:, 1]
    las.z = xyz[:, 2]
    nx, ny, nz = _encode_normals(
        normals
    )
    las["normal x"] = nx
    las["normal y"] = ny
    las["normal z"] = nz
    las.classification = np.zeros(
        xyz.shape[0],
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
            threshold=0.50,
            scalar=1.25,
            chunk_size=5000,
            terrain3d_enabled=True,
            terrain3d_voxel=0.50,
            terrain3d_surface_thickness=0.22,
            terrain3d_coherence=0.62,
            terrain3d_max_normal_angle_deg=88.0,
        ),
    )

    classes = result.model.classify_xyz(
        np.asarray(cloud.las.x),
        np.asarray(cloud.las.y),
        np.asarray(cloud.las.z),
    )
    terrain_classes = classes[
        : terrain.shape[0]
    ]
    vegetation_classes = classes[
        terrain.shape[0] :
    ]

    assert result.terrain3d_voxel_count > 0
    assert (
        np.mean(
            terrain_classes
            == GROUND_CLASS
        )
        > 0.95
    )
    assert (
        np.mean(
            vegetation_classes
            == NON_GROUND_CLASS
        )
        > 0.95
    )
