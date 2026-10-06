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
    assert result["reconstructed_is_measured_ground"] is False
    assert result["interpolated_is_measured_ground"] is False


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


def test_mdt_marks_reconstructed_ground_separately(tmp_path):
    source = tmp_path / "tiny_complete.las"
    header = laspy.LasHeader(point_format=3, version="1.2")
    header.add_crs(CRS.from_epsg(3763))
    cloud = laspy.LasData(header)
    cloud.x = np.array([0.0, 1.0], dtype=np.float64)
    cloud.y = np.array([0.0, 0.0], dtype=np.float64)
    cloud.z = np.array([10.0, 10.0], dtype=np.float64)
    cloud.write(source)

    class CompleteModel:
        params = SimpleNamespace(chunk_size=1000)

        @staticmethod
        def classify_points(points, x, y, z):
            return np.array([2] * len(x), dtype=np.uint8)

        @staticmethod
        def iter_synthetic_fill_xyz():
            yield (
                np.array([0.5], dtype=np.float64),
                np.array([0.0], dtype=np.float64),
                np.array([10.0], dtype=np.float64),
            )

    preview = build_ground_mdt_preview(
        source,
        CompleteModel(),
        resolution_m=0.5,
        max_gap_m=0.0,
    )
    assert preview.reconstructed_points == 1
    assert 2 in set(preview.state.ravel().tolist())


def test_r20_6_2_fast_mdt_uses_solved_model_without_reclassifying_source(tmp_path):
    source = tmp_path / "header_only.las"
    header = laspy.LasHeader(point_format=3, version="1.2")
    header.add_crs(CRS.from_epsg(3763))
    cloud = laspy.LasData(header)
    cloud.x = np.array([1000.0, 1002.0], dtype=np.float64)
    cloud.y = np.array([2000.0, 2002.0], dtype=np.float64)
    cloud.z = np.array([100.0, 102.0], dtype=np.float64)
    cloud.write(source)

    shape = (4, 4)
    mantle_surface = (
        100.0
        + np.arange(4, dtype=np.float32)[:, None] * 0.2
        + np.arange(4, dtype=np.float32)[None, :] * 0.1
    )
    measured = np.ones(shape, dtype=bool)
    measured[1:3, 1:3] = False
    fill = np.zeros(shape, dtype=bool)
    fill[1:3, 1:3] = True
    reconstruction_surface = mantle_surface.copy()
    reconstruction_surface[1:3, 1:3] -= 0.3

    model = SimpleNamespace(
        mantle=SimpleNamespace(
            origin=np.array([1000.0, 2000.0]),
            cell_size=0.5,
            nx=4,
            ny=4,
            surface=mantle_surface,
        ),
        reconstruction=SimpleNamespace(
            fill_mask=fill,
            surface=reconstruction_surface,
        ),
        measured_ground_cells=measured,
        measured_ground_point_count=123,
        synthetic_fill_point_count=16,
    )

    # Deliberately no classify_points/evaluate_points/classify_xyz method.
    preview = build_ground_mdt_preview(
        source,
        model,
        resolution_m=0.25,
        max_gap_m=0.0,
    )

    assert preview.ground_points == 123
    assert preview.reconstructed_points == 16
    states = set(preview.state.ravel().tolist())
    assert 1 in states
    assert 2 in states
    assert 3 not in states
    assert np.isfinite(preview.elevation[preview.state > 0]).all()
