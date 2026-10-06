from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from las_classifier.classifiers import universal_ground
from las_classifier.terrain.elevated_surface_guard import (
    ElevatedSurfaceGuard,
    apply_elevated_surface_veto,
)
from las_classifier.terrain.ground_evidence import (
    GroundDecision,
    PROV_ELEVATED_SURFACE_VETO,
)
from las_classifier.terrain.mantle_veto import MantleVeto, MantleVetoConfig


def _base_guard() -> MantleVeto:
    shape = (2, 2)
    return MantleVeto(
        origin=np.array([0.0, 0.0], dtype=np.float64),
        cell_size=1.0,
        nx=2,
        ny=2,
        surface=np.zeros(shape, dtype=np.float32),
        reliable=np.ones(shape, dtype=bool),
        slope_x=np.zeros(shape, dtype=np.float32),
        slope_y=np.zeros(shape, dtype=np.float32),
        roof_candidate=np.array([[False, True], [False, False]], dtype=bool),
        canopy_candidate=np.zeros(shape, dtype=bool),
        directional_lower_count=np.zeros(shape, dtype=np.uint8),
        config=MantleVetoConfig(),
    )


def test_r2051_continuity_does_not_mutate_mantle_guard(monkeypatch):
    guard = _base_guard()
    original_roof = guard.roof_candidate.copy()
    mantle = SimpleNamespace(veto_guard=guard)
    extra = ElevatedSurfaceGuard(
        origin=np.array([0.0, 0.0], dtype=np.float64),
        cell_size=1.0,
        nx=2,
        ny=2,
        blocked=np.array([[True, False], [False, False]], dtype=bool),
    )
    captured = {}

    monkeypatch.setattr(
        universal_ground,
        "build_elevated_surface_guard",
        lambda context, mantle: extra,
    )

    def fake_build(context, received_mantle, combined_guard):
        captured["guard"] = combined_guard
        assert received_mantle is mantle
        return SimpleNamespace(final_veto=None)

    monkeypatch.setattr(
        universal_ground,
        "build_ground_continuity",
        fake_build,
    )

    result = universal_ground._r2051_continuity_builder(
        object(), mantle, None
    )

    # Critical regression: R20.5.1 may not rewrite MANTO R20.4 diagnostics.
    np.testing.assert_array_equal(guard.roof_candidate, original_roof)
    assert mantle.veto_guard is guard

    combined = captured["guard"]
    assert combined is not guard
    np.testing.assert_array_equal(
        combined.roof_candidate,
        np.array([[True, True], [False, False]], dtype=bool),
    )
    assert result.final_veto is extra


class _Evidence:
    def __init__(self):
        self.decision = np.array(
            [
                int(GroundDecision.L3_GROUND_ORIGINAL_VALIDATED),
                int(GroundDecision.NON_GROUND_VEGETATION),
            ],
            dtype=np.uint8,
        )
        self.provenance = np.zeros(2, dtype=np.uint16)
        self.score = np.ones(2, dtype=np.float32)

    def classifications(self):
        return np.where(
            self.decision == int(GroundDecision.L3_GROUND_ORIGINAL_VALIDATED),
            2,
            1,
        ).astype(np.uint8)


def test_final_veto_rejects_ground_without_touching_already_non_ground():
    evidence = _Evidence()
    guard = ElevatedSurfaceGuard(
        origin=np.array([0.0, 0.0], dtype=np.float64),
        cell_size=1.0,
        nx=1,
        ny=1,
        blocked=np.array([[True]], dtype=bool),
    )
    x = np.array([0.5, 0.5])
    y = np.array([0.5, 0.5])
    z = np.array([10.0, 11.0])
    applied = apply_elevated_surface_veto(
        evidence, guard, x, y, z, np.array([False, False])
    )

    np.testing.assert_array_equal(applied, [True, False])
    assert evidence.decision[0] == int(GroundDecision.NON_GROUND_OBJECT)
    assert evidence.decision[1] == int(GroundDecision.NON_GROUND_VEGETATION)
    assert evidence.provenance[0] & PROV_ELEVATED_SURFACE_VETO
