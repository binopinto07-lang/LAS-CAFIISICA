from __future__ import annotations

import numpy as np

from las_classifier.classifiers.hybrid_ground import (
    HybridGroundModel,
    _terrain3d_params,
)
from las_classifier.ground.types import GroundEngineParams


class _PTD:
    synthetic_fill_point_count = 0
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
        if False:
            yield None

    def iter_viewer_synthetic_fill_xyz(self):
        if False:
            yield None


class _CSF:
    def confidence_xyz(self, x, y, z):
        return np.zeros(x.shape[0], dtype=np.float64)


class _Terrain3D:
    terrain_voxel_count = 1
    seed_voxels = 1

    def ground_mask(self, x, y, z):
        return np.asarray(
            [True, False],
            dtype=np.bool_,
        )


def test_hybrid_3d_surface_can_rescue_point_rejected_by_2d_engines():
    model = HybridGroundModel(
        params=GroundEngineParams(
            confidence_threshold=0.66,
        ),
        ptd=_PTD(),
        csf=_CSF(),
        terrain3d=_Terrain3D(),
    )
    x = np.array([1.0, 2.0])
    y = np.array([1.0, 2.0])
    z = np.array([5.0, 5.0])

    classes = model.classify_xyz(x, y, z)

    assert int(classes[0]) == 2
    assert int(classes[1]) == 1


def test_extreme_hybrid_uses_finer_true_3d_surface():
    params = _terrain3d_params(
        GroundEngineParams.preset("extreme")
    )

    assert params.voxel == 0.35
    assert params.surface_thickness == 0.12
    assert params.target_sample_points == 30_000_000
