from __future__ import annotations

import numpy as np

from las_classifier.ground.terrain_tin import TerrainTIN


def test_tin_handles_large_epsg_coordinates_and_near_degenerate_rows():
    x = np.linspace(55488.783, 55918.072, 1200)
    rows = []
    for dy in (0.0, 0.00002, 0.00004, 0.00006):
        y = np.full_like(x, 162281.029 + dy)
        z = 177.0 + 0.12 * (x - x.min()) + 0.03 * dy
        rows.append(np.column_stack((x, y, z)))

    vertices = np.vstack(rows)
    vertices = np.vstack((vertices, vertices[::100]))
    tin = TerrainTIN.build(vertices)

    assert tin.triangle_count > 0
    assert tin.vertices.shape[0] < vertices.shape[0]

    qx = np.array([55600.0, 55800.0])
    qy = np.array([162281.02903, 162281.02903])
    qz = 177.0 + 0.12 * (qx - 55488.783)
    metrics = tin.metrics(qx, qy, qz)

    assert np.all(metrics["valid"])
    assert np.all(metrics["plane_distance"] < 0.01)


def test_tin_query_uses_same_normalized_coordinate_system():
    x = np.linspace(55488.0, 55918.0, 32)
    y = np.linspace(162281.0, 162949.0, 32)
    xx, yy = np.meshgrid(x, y)
    zz = 200.0 + 0.02 * xx + 0.01 * yy
    vertices = np.column_stack((xx.ravel(), yy.ravel(), zz.ravel()))

    tin = TerrainTIN.build(vertices)
    qx = np.array([55600.0, 55750.0])
    qy = np.array([162500.0, 162700.0])
    qz = 200.0 + 0.02 * qx + 0.01 * qy
    metrics = tin.metrics(qx, qy, qz)

    assert np.all(metrics["valid"])
    assert np.max(metrics["plane_distance"]) < 1e-6
