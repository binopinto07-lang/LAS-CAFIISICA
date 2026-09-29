from __future__ import annotations

import laspy
import numpy as np

from las_classifier.cloud.loader import load_cloud
from las_classifier.terrain.l3_recovery import (
    L3RecoveryParams,
    build_l3_recovery,
)
from las_classifier.terrain.schema import (
    SourceInspection,
    SourceType,
)


class _PTD:
    def _confidence(self, x, y, z, points=None):
        count = x.size
        score = np.full(count, 0.80, dtype=np.float64)
        metrics = {
            "valid": np.ones(count, dtype=np.bool_),
            "plane_distance": np.full(
                count,
                0.05,
                dtype=np.float64,
            ),
        }
        return score, metrics


def _inspection(source_type: SourceType) -> SourceInspection:
    return SourceInspection(
        source_type=source_type,
        confidence=1.0,
        evidence=("test",),
        point_format_id=3,
        generating_software="DJI Terra",
        system_identifier="",
        has_rgb=True,
        has_gps_time=True,
        has_intensity=True,
        has_scan_angle=True,
        has_returns=True,
        max_return_number=2,
        max_number_of_returns=2,
        multi_return_fraction=1.0,
        last_return_fraction=0.75,
        only_return_fraction=0.0,
        sample_count=4,
    )


def _cloud(path):
    header = laspy.LasHeader(
        point_format=3,
        version="1.2",
    )
    header.generating_software = "DJI Terra"
    las = laspy.LasData(header)
    las.x = np.array([10.00, 10.05, 10.10, 10.15])
    las.y = np.array([20.00, 20.04, 20.08, 20.12])
    las.z = np.array([100.00, 100.02, 100.04, 100.06])
    las.classification = np.array([2, 1, 1, 1], dtype=np.uint8)
    las.return_number = np.array([2, 2, 2, 1], dtype=np.uint8)
    las.number_of_returns = np.array([2, 2, 2, 2], dtype=np.uint8)
    las.write(path)
    return load_cloud(path)


def test_l3_recovery_uses_return_plus_geometry_not_return_alone(tmp_path):
    cloud = _cloud(tmp_path / "l3.las")
    model = build_l3_recovery(
        cloud,
        _PTD(),
        _inspection(SourceType.L3_LIDAR),
        params=L3RecoveryParams(
            sample_target=100,
            voxel_size=0.50,
            max_plane_distance=0.25,
            min_points_per_voxel=2,
            min_seed_fraction=0.05,
            min_geometry_score=0.30,
            strong_geometry_score=0.60,
        ),
    )

    assert model is not None
    points = cloud.las.points
    mask = model.recovered_mask(
        points,
        np.asarray(cloud.las.x),
        np.asarray(cloud.las.y),
        np.asarray(cloud.las.z),
    )

    assert mask.tolist() == [True, True, True, False]


def test_l3_recovery_is_disabled_for_p1(tmp_path):
    cloud = _cloud(tmp_path / "p1.las")
    model = build_l3_recovery(
        cloud,
        _PTD(),
        _inspection(SourceType.P1_PHOTOGRAMMETRY),
    )

    assert model is None
