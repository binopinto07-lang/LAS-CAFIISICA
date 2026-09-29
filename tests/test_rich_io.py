from __future__ import annotations

import laspy
import numpy as np

from las_classifier.cloud.loader import load_cloud
from las_classifier.terrain.rich_io import iter_rich_chunks


def test_rich_chunk_preserves_las_evidence(tmp_path):
    path = tmp_path / "rich.las"
    header = laspy.LasHeader(
        point_format=3,
        version="1.2",
    )
    las = laspy.LasData(header)
    las.x = np.array([1.0, 2.0, 3.0])
    las.y = np.array([4.0, 5.0, 6.0])
    las.z = np.array([7.0, 8.0, 9.0])
    las.classification = np.array([2, 1, 7], dtype=np.uint8)
    las.return_number = np.array([1, 2, 1], dtype=np.uint8)
    las.number_of_returns = np.array([2, 2, 1], dtype=np.uint8)
    las.intensity = np.array([10, 20, 30], dtype=np.uint16)
    las.gps_time = np.array([100.0, 101.0, 102.0])
    las.point_source_id = np.array([4, 5, 6], dtype=np.uint16)
    las.red = np.array([100, 200, 300], dtype=np.uint16)
    las.green = np.array([400, 500, 600], dtype=np.uint16)
    las.blue = np.array([700, 800, 900], dtype=np.uint16)
    las.write(path)

    cloud = load_cloud(path)
    chunk = next(iter_rich_chunks(cloud, chunk_size=3))

    assert chunk.point_count == 3
    assert chunk.xyz.shape == (3, 3)
    assert chunk.return_number.tolist() == [1, 2, 1]
    assert chunk.number_of_returns.tolist() == [2, 2, 1]
    assert chunk.intensity.tolist() == [10, 20, 30]
    assert chunk.gps_time.tolist() == [100.0, 101.0, 102.0]
    assert chunk.point_source_id.tolist() == [4, 5, 6]
    assert chunk.red.tolist() == [100, 200, 300]
