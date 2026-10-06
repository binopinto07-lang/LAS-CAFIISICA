"""R20.5 preview must never write raster before user explicitly exports it."""
from types import SimpleNamespace

import laspy
import numpy as np
import pytest
from pyproj import CRS

from las_classifier.terrain.mdt_export import (
    build_ground_mdt_preview,
    export_ground_mdt,
    fill_small_mdt_gaps,
    MDTPreview,
)


def test_preview_is_in_memory_and_user_export_is_separate(tmp_path):
    source = tmp_path / "tiny_ground.las"
    header = laspy.LasHeader(point_format=3, version="1.2")
    header.add_crs(CRS.from_epsg(3763))
    cloud = laspy.LasData(header)
    cloud.x = np.array([0, 0.5, 1.0, 0, 0.5, 1.0], dtype=np.float64)
    cloud.y = np.array([0, 0, 0, 1, 1, 1], dtype=np.float64)
    cloud.z = np.array([10, 10.1, 10.2, 10, 13, 10.2], dtype=np.float64)
    cloud.write(source)

    class TestModel:
        params = SimpleNamespace(chunk_size=1000)

        @staticmethod
        def classify_points(points, x, y, z):
            return np.where(z > 12, 1, 2).astype(np.uint8)

    model = TestModel()
    preview = build_ground_mdt_preview(
        source, model, resolution_m=0.5, max_gap_m=0.5,
    )
    assert isinstance(preview, MDTPreview)
    assert preview.ground_points == 5
    assert not list(tmp_path.glob("*.tif"))
    assert not list(tmp_path.glob("*.json"))
    payload = preview.browser_payload(max_side=256)
    assert payload["width"] >= 2
    assert len(payload["z"]) == payload["width"] * payload["height"]
    assert set(payload["state"]) <= {0, 1, 2, 3}

    path = tmp_path / "ground_mdt.tif"
    result = export_ground_mdt(
        source, path, model,
        preview=preview,
        resolution_m=0.5,
        max_gap_m=0.5,
    )
    assert path.exists()
    assert (tmp_path / "ground_mdt_OBSERVATION_STATE.tif").exists()
    assert (tmp_path / "ground_mdt.json").exists()
    assert result["ground_points"] == 5
    assert result["reconstructed_is_measured_ground"] is False\n    assert result["interpolated_is_measured_ground"] is False


def test_mdt_raster_does_not_turn_unknown_terrain_into_measured_ground():
    surface = np.full((7, 7), np.nan, dtype=np.float32)
    observed = np.zeros((7, 7), dtype=bool)
    observed[3, 3] = True
    surface[3, 3] = 20.0
    filled, state = fill_small_mdt_gaps(
        surface, observed, resolution_m=0.25, max_gap_m=0.50,
    )
    assert state[3, 3] == 1
    assert state[3, 4] == 3
    assert state[0, 0] == 0
    assert np.isnan(filled[0, 0])


def test_cannot_export_a_preview_from_another_source(tmp_path):
    preview = MDTPreview(
        source=tmp_path / "one.las",
        elevation=np.ones((2, 2), dtype=np.float32),
        state=np.ones((2, 2), dtype=np.uint8),
        xmin=0, ymax=1, resolution_m=0.25, max_gap_m=0.5,
        ground_points=4,
    )
    with pytest.raises(ValueError, match="another LAS"):
        export_ground_mdt(tmp_path / "other.las", tmp_path / "out.tif", None, preview=preview)
