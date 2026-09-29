from __future__ import annotations

import numpy as np

from las_classifier.classifiers.hybrid_ground import (
    HybridGroundModel,
    _terrain3d_params,
)
from las_classifier.ground.types import GroundEngineParams
from las_classifier.terrain.schema import (
    SourceInspection,
    SourceType,
)


class _PTD:
    synthetic_fill_point_count = 100
    effective_fill_spacing = 0.05

    def _confidence(self, x, y, z, points=None):
        score = np.zeros(x.shape[0], dtype=np.float64)
        metrics = {
            "plane_distance": np.full(
                x.shape[0],
                1.0,
                dtype=np.float64,
            )
        }
        return score, metrics

    def iter_synthetic_fill_xyz(self):
        yield (
            np.array([0.0]),
            np.array([0.0]),
            np.array([0.0]),
        )

    def iter_viewer_synthetic_fill_xyz(self):
        yield from self.iter_synthetic_fill_xyz()


class _CSF:
    def confidence_xyz(self, x, y, z):
        return np.zeros(x.shape[0], dtype=np.float64)


class _SMRF:
    def classify_xyz(self, x, y, z):
        return np.ones(x.shape[0], dtype=np.uint8)


class _Terrain3D:
    terrain_voxel_count = 1
    seed_voxels = 1

    def ground_mask(self, x, y, z):
        return np.ones(
            x.shape[0],
            dtype=np.bool_,
        )


def _p1_inspection():
    return SourceInspection(
        source_type=SourceType.P1_PHOTOGRAMMETRY,
        confidence=1.0,
        evidence=("test",),
        point_format_id=3,
        generating_software="Agisoft Metashape",
        system_identifier="",
        has_rgb=True,
        has_gps_time=False,
        has_intensity=True,
        has_scan_angle=True,
        has_returns=True,
        max_return_number=1,
        max_number_of_returns=1,
        multi_return_fraction=0.0,
        last_return_fraction=1.0,
        only_return_fraction=1.0,
        sample_count=1,
    )


def test_p1_terrain3d_is_support_gate_not_positive_rescue():
    model = HybridGroundModel(
        params=GroundEngineParams(
            confidence_threshold=0.66,
        ),
        ptd=_PTD(),
        csf=_CSF(),
        smrf=_SMRF(),
        terrain3d=_Terrain3D(),
        source_inspection=_p1_inspection(),
    )
    x = np.array([1.0, 2.0])
    y = np.array([1.0, 2.0])
    z = np.array([5.0, 5.0])

    classes = model.classify_xyz(x, y, z)

    assert np.all(classes == 1)


def test_p1_synthetic_fill_disabled_until_observability_model():
    model = HybridGroundModel(
        params=GroundEngineParams(),
        ptd=_PTD(),
        csf=_CSF(),
        smrf=_SMRF(),
        source_inspection=_p1_inspection(),
    )

    assert model.synthetic_fill_point_count == 0
    assert list(model.iter_synthetic_fill_xyz()) == []


def test_extreme_p1_support_surface_is_strict():
    params = _terrain3d_params(
        GroundEngineParams.preset("extreme")
    )

    assert params.voxel == 0.30
    assert params.surface_thickness == 0.09
    assert params.coherence == 0.80
    assert params.target_sample_points == 30_000_000
