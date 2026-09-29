from __future__ import annotations

import numpy as np

from las_classifier.classifiers.adaptive_ptd import (
    AdaptivePTDModel,
    GROUND_CLASS,
    NON_GROUND_CLASS,
)
from las_classifier.ground.gap_detection import GapAnalysis
from las_classifier.ground.terrain_tin import TerrainTIN
from las_classifier.ground.types import (
    GroundAnalysis,
    GroundEngineParams,
)


class _PointFormat:
    dimension_names = (
        "normal x",
        "normal y",
        "normal z",
    )


class _FakePoints:
    point_format = _PointFormat()

    def __init__(self, count: int) -> None:
        self._values = {
            "normal x": np.zeros(count, dtype=np.float64),
            "normal y": np.ones(count, dtype=np.float64),
            "normal z": np.zeros(count, dtype=np.float64),
        }

    def __getitem__(self, key):
        return self._values[key]


def _model() -> AdaptivePTDModel:
    vertices = np.array(
        [
            [0.0, 0.0, 100.0],
            [2.0, 0.0, 100.0],
            [0.0, 2.0, 100.0],
            [2.0, 2.0, 100.0],
        ],
        dtype=np.float64,
    )
    tin = TerrainTIN.build(vertices)
    gaps = GapAnalysis(
        kind=np.zeros(tin.triangle_count, dtype=np.uint8),
        supported_mask=np.zeros(
            tin.triangle_count,
            dtype=np.bool_,
        ),
        detected_count=0,
        supported_count=0,
        occluded_count=0,
        rejected_count=0,
    )
    return AdaptivePTDModel(
        params=GroundEngineParams(
            confidence_threshold=0.66,
        ),
        analysis=GroundAnalysis(
            point_count=4,
            median_spacing=0.03,
            xy_density=1000.0,
            z_range=1.0,
            sample_stride=1,
            sample_count=4,
        ),
        tin=tin,
        discontinuity=np.zeros(
            tin.triangle_count,
            dtype=np.float64,
        ),
        gaps=gaps,
        distance_limit=0.18,
        max_triangle_edge=4.0,
        effective_fill_spacing=0.05,
        synthetic_fill_point_count=0,
    )


def test_point_within_ten_cm_is_ground_even_with_bad_normal():
    model = _model()
    x = np.array([1.0])
    y = np.array([1.0])
    z = np.array([100.08])

    classes = model.classify_points(
        _FakePoints(1),
        x,
        y,
        z,
    )

    assert int(classes[0]) == int(GROUND_CLASS)


def test_point_above_ten_cm_is_not_forced_to_ground():
    model = _model()
    x = np.array([1.0])
    y = np.array([1.0])
    z = np.array([100.16])

    classes = model.classify_points(
        _FakePoints(1),
        x,
        y,
        z,
    )

    assert int(classes[0]) == int(NON_GROUND_CLASS)


def test_steep_talude_uses_true_point_to_plane_ten_cm_rule():
    slope = 1.7320508075688772
    vertices = np.array(
        [
            [0.0, 0.0, 100.0],
            [2.0, 0.0, 100.0 + 2.0 * slope],
            [0.0, 2.0, 100.0],
            [2.0, 2.0, 100.0 + 2.0 * slope],
        ],
        dtype=np.float64,
    )
    tin = TerrainTIN.build(vertices)
    gaps = GapAnalysis(
        kind=np.zeros(tin.triangle_count, dtype=np.uint8),
        supported_mask=np.zeros(tin.triangle_count, dtype=np.bool_),
        detected_count=0,
        supported_count=0,
        occluded_count=0,
        rejected_count=0,
    )
    model = AdaptivePTDModel(
        params=GroundEngineParams(confidence_threshold=0.66),
        analysis=GroundAnalysis(
            point_count=4,
            median_spacing=0.03,
            xy_density=1000.0,
            z_range=5.0,
            sample_stride=1,
            sample_count=4,
        ),
        tin=tin,
        discontinuity=np.zeros(tin.triangle_count, dtype=np.float64),
        gaps=gaps,
        distance_limit=0.18,
        max_triangle_edge=4.0,
        effective_fill_spacing=0.03,
        synthetic_fill_point_count=0,
    )

    x = np.array([1.0])
    y = np.array([1.0])
    terrain_z = 100.0 + slope
    z = np.array([terrain_z + 0.15])

    metrics = tin.metrics(x, y, z)
    assert float(metrics["vertical_residual"][0]) > 0.10
    assert float(metrics["plane_distance"][0]) < 0.10

    classes = model.classify_points(_FakePoints(1), x, y, z)
    assert int(classes[0]) == int(GROUND_CLASS)
