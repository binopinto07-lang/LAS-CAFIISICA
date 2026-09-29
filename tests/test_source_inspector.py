from __future__ import annotations

import laspy
import numpy as np

from las_classifier.cloud.loader import load_cloud
from las_classifier.terrain.schema import SourceType
from las_classifier.terrain.source_inspector import inspect_source


def _write_cloud(
    path,
    *,
    producer: str,
    return_number,
    number_of_returns,
):
    header = laspy.LasHeader(
        point_format=3,
        version="1.2",
    )
    header.generating_software = producer
    las = laspy.LasData(header)
    count = len(return_number)
    las.x = np.linspace(0.0, 1.0, count)
    las.y = np.linspace(0.0, 1.0, count)
    las.z = np.linspace(10.0, 11.0, count)
    las.return_number = np.asarray(
        return_number,
        dtype=np.uint8,
    )
    las.number_of_returns = np.asarray(
        number_of_returns,
        dtype=np.uint8,
    )
    las.red = np.full(count, 1000, dtype=np.uint16)
    las.green = np.full(count, 2000, dtype=np.uint16)
    las.blue = np.full(count, 3000, dtype=np.uint16)
    las.write(path)


def test_metashape_only_returns_is_detected_as_p1(tmp_path):
    path = tmp_path / "p1.las"
    _write_cloud(
        path,
        producer="Agisoft Metashape",
        return_number=[1] * 100,
        number_of_returns=[1] * 100,
    )
    inspection = inspect_source(load_cloud(path))

    assert inspection.source_type is SourceType.P1_PHOTOGRAMMETRY
    assert inspection.confidence >= 0.95
    assert inspection.multi_return_fraction == 0.0
    assert inspection.only_return_fraction == 1.0


def test_dji_terra_real_multi_returns_is_detected_as_l3(tmp_path):
    path = tmp_path / "l3.las"
    rn = [1, 2, 1, 2] * 50
    nr = [2, 2, 2, 2] * 50
    _write_cloud(
        path,
        producer="DJI Terra",
        return_number=rn,
        number_of_returns=nr,
    )
    inspection = inspect_source(load_cloud(path))

    assert inspection.source_type is SourceType.L3_LIDAR
    assert inspection.confidence >= 0.95
    assert inspection.max_number_of_returns == 2
    assert inspection.multi_return_fraction == 1.0


def test_manual_source_override_wins(tmp_path):
    path = tmp_path / "manual.las"
    _write_cloud(
        path,
        producer="Unknown",
        return_number=[1] * 10,
        number_of_returns=[1] * 10,
    )
    inspection = inspect_source(
        load_cloud(path),
        override=SourceType.L3_LIDAR,
    )

    assert inspection.source_type is SourceType.L3_LIDAR
    assert inspection.confidence == 1.0
