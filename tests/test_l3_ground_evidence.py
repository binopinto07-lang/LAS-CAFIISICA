from __future__ import annotations

import laspy
import numpy as np

from las_classifier.classifiers.l3_ground_evidence import (
    L3EvidenceParams,
    L3GroundEvidenceModel,
)
from las_classifier.ground.types import GroundEngineParams
from las_classifier.terrain.schema import (
    SourceInspection,
    SourceType,
)


class _PTD:
    max_triangle_edge = 2.0

    def __init__(self, score: float) -> None:
        self.score = float(score)

    def _confidence(self, x, y, z, points=None):
        count = x.size
        score = np.full(count, self.score, dtype=np.float64)
        metrics = {
            "valid": np.ones(count, dtype=np.bool_),
            "vertical_residual": np.zeros(count, dtype=np.float64),
            "plane_distance": np.full(count, 0.03, dtype=np.float64),
            "max_edge": np.full(count, 0.5, dtype=np.float64),
            "normal": np.tile(
                np.array([[0.0, 0.0, 1.0]], dtype=np.float64),
                (count, 1),
            ),
        }
        return score, metrics


class _Cloth:
    def __init__(self, score: float) -> None:
        self.score = float(score)

    def confidence_xyz(self, x, y, z):
        return np.full(x.size, self.score, dtype=np.float64)


class _SMRF:
    def __init__(self, ground: bool) -> None:
        self.ground = bool(ground)

    def classify_xyz(self, x, y, z):
        value = 2 if self.ground else 1
        return np.full(x.size, value, dtype=np.uint8)


class _Support:
    def __init__(self, score: float) -> None:
        self.score = float(score)

    def score_xyz(self, x, y, z):
        return np.full(x.size, self.score, dtype=np.float64)


def _inspection() -> SourceInspection:
    return SourceInspection(
        source_type=SourceType.L3_LIDAR,
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
        max_return_number=4,
        max_number_of_returns=4,
        multi_return_fraction=1.0,
        last_return_fraction=0.5,
        only_return_fraction=0.0,
        sample_count=2,
    )


def _points(classification: int):
    header = laspy.LasHeader(
        point_format=3,
        version="1.2",
    )
    las = laspy.LasData(header)
    las.x = np.array([1.0])
    las.y = np.array([2.0])
    las.z = np.array([3.0])
    las.classification = np.array(
        [classification],
        dtype=np.uint8,
    )
    las.return_number = np.array([4], dtype=np.uint8)
    las.number_of_returns = np.array([4], dtype=np.uint8)
    return las.points


def _model(
    *,
    ptd: float,
    cloth: float,
    support: float,
    smrf_ground: bool,
) -> L3GroundEvidenceModel:
    return L3GroundEvidenceModel(
        params=GroundEngineParams(),
        evidence_params=L3EvidenceParams(
            minimum_ground_score=0.68,
            high_confidence_score=0.82,
        ),
        ptd=_PTD(ptd),
        cloth=_Cloth(cloth),
        smrf=_SMRF(smrf_ground),
        support=_Support(support),
        source_inspection=_inspection(),
    )


def test_last_return_alone_never_becomes_ground():
    model = _model(
        ptd=0.05,
        cloth=0.05,
        support=0.0,
        smrf_ground=False,
    )
    points = _points(1)
    x = np.array([1.0])
    y = np.array([2.0])
    z = np.array([3.0])

    evidence = model.evidence(points, x, y, z)
    classes = model.classify_points(points, x, y, z)

    assert evidence["return_score"][0] == 1.0
    assert not bool(evidence["ground"][0])
    assert int(classes[0]) == 1


def test_geometry_support_and_last_return_can_recover_ground():
    model = _model(
        ptd=0.92,
        cloth=0.88,
        support=0.90,
        smrf_ground=True,
    )
    points = _points(1)
    x = np.array([1.0])
    y = np.array([2.0])
    z = np.array([3.0])

    evidence = model.evidence(points, x, y, z)
    classes = model.classify_points(points, x, y, z)

    assert evidence["score"][0] >= 0.68
    assert bool(evidence["ground"][0])
    assert int(classes[0]) == 2


def test_original_classification_does_not_change_decision():
    model = _model(
        ptd=0.85,
        cloth=0.80,
        support=0.80,
        smrf_ground=True,
    )
    x = np.array([1.0])
    y = np.array([2.0])
    z = np.array([3.0])

    class1 = model.classify_points(
        _points(1),
        x,
        y,
        z,
    )
    class2 = model.classify_points(
        _points(2),
        x,
        y,
        z,
    )
    class7 = model.classify_points(
        _points(7),
        x,
        y,
        z,
    )

    assert int(class1[0]) == int(class2[0]) == int(class7[0]) == 2
