from __future__ import annotations

from las_classifier.classifiers.adaptive_ptd import _adaptive_settings
from las_classifier.ground.types import GroundAnalysis, GroundEngineParams


def test_extreme_reconstruction_tracks_native_xy_spacing():
    analysis = GroundAnalysis(
        point_count=318_154_588,
        median_spacing=0.060,
        xy_density=1109.1536381518767,
        z_range=109.766,
        sample_stride=46,
        sample_count=6_916_405,
    )
    params = GroundEngineParams.preset("extreme")

    _, _, _, _, synthetic_spacing = _adaptive_settings(
        analysis,
        params,
    )

    assert 0.025 <= synthetic_spacing <= 0.032


def test_high_quality_reconstruction_is_denser_than_old_sample_spacing():
    analysis = GroundAnalysis(
        point_count=318_154_588,
        median_spacing=0.060,
        xy_density=1109.1536381518767,
        z_range=109.766,
        sample_stride=46,
        sample_count=6_916_405,
    )
    params = GroundEngineParams.preset("high")

    _, _, _, _, synthetic_spacing = _adaptive_settings(
        analysis,
        params,
    )

    assert synthetic_spacing < analysis.median_spacing
