"""R20.4 universal-source policy regression tests."""
from __future__ import annotations

import numpy as np

from las_classifier.classifiers import universal_ground
from las_classifier.terrain.ground_evidence import GroundEvidenceScorer


def test_single_return_metadata_is_neutral_when_not_informative():
    rn = np.ones(8, dtype=np.uint8)
    nr = np.ones(8, dtype=np.uint8)
    score = GroundEvidenceScorer._return_score(
        rn, nr, len(rn), enabled=False,
    )
    np.testing.assert_allclose(score, 0.5)


def test_real_multi_return_can_still_be_optional_evidence():
    rn = np.array([1, 2, 1], dtype=np.uint8)
    nr = np.array([2, 2, 1], dtype=np.uint8)
    score = GroundEvidenceScorer._return_score(
        rn, nr, len(rn), enabled=True,
    )
    np.testing.assert_allclose(score, [0.35, 1.0, 1.0])


def test_universal_wrapper_never_requires_lidar(monkeypatch):
    captured = {}

    def fake_run(cloud, params, progress, **kwargs):
        captured.update(kwargs)
        return "ok"

    monkeypatch.setattr(universal_ground, "run_l3_ground_lab", fake_run)
    assert universal_ground.run_universal_ground(object()) == "ok"
    assert captured["require_lidar"] is False
    assert captured["source_override"] is None
    assert captured["revision_label"] == "R20.5"


def test_r204_comparator_keeps_previous_revision(monkeypatch):
    captured = {}

    def fake_run(cloud, params, progress, **kwargs):
        captured.update(kwargs)
        return "ok"

    monkeypatch.setattr(universal_ground, "run_l3_ground_lab", fake_run)
    assert universal_ground.run_universal_ground_r204(object()) == "ok"
    assert captured["require_lidar"] is False
    assert captured["source_override"] is None
    assert captured["revision_label"] == "R20.4"
