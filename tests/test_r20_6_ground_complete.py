from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from las_classifier.classifiers.universal_ground import UniversalCompleteGroundModel
from las_classifier.ground.types import GroundEngineParams
from las_classifier.terrain.mantle_reconstruction import (
    build_mantle_ground_reconstruction,
)


def _mantle():
    shape = (5, 5)
    surface = np.full(shape, 100.0, dtype=np.float32)
    surface[:, :] += np.arange(5, dtype=np.float32)[None, :] * 0.2
    reliable = np.ones(shape, dtype=bool)
    inferred = np.zeros(shape, dtype=bool)
    hidden = np.zeros(shape, dtype=bool)

    # One empty cell and one "upper returns but Ground not observed" cell.
    inferred[2, 2] = True
    hidden[3, 2] = True
    reliable[inferred | hidden] = False

    guard = SimpleNamespace(
        roof_candidate=np.zeros(shape, dtype=bool),
        canopy_candidate=np.zeros(shape, dtype=bool),
    )
    return SimpleNamespace(
        origin=np.array([1000.0, 2000.0]),
        cell_size=0.5,
        nx=5,
        ny=5,
        surface=surface,
        observed=~inferred,
        reliable=reliable,
        inferred=inferred,
        ambiguous=~reliable,
        possible_no_ground_observation=hidden,
        anchor_distance=np.zeros(shape, dtype=np.float32),
        slope_x=np.full(shape, 0.4, dtype=np.float32),
        slope_y=np.zeros(shape, dtype=np.float32),
        breakline=np.zeros(shape, dtype=bool),
        veto_guard=guard,
    )


def test_r20_6_reconstructs_only_explicit_no_ground_observation_cells():
    mantle = _mantle()
    reconstruction = build_mantle_ground_reconstruction(
        mantle, spacing_m=0.25, max_points=1000,
    )

    assert reconstruction.fill_cell_count == 2
    assert reconstruction.inferred_cell_count == 1
    assert reconstruction.hidden_ground_cell_count == 1
    assert reconstruction.point_count == 8
    assert reconstruction.source_state[2, 2] == 1
    assert reconstruction.source_state[3, 2] == 2
    assert not reconstruction.fill_mask[0, 0]

    xyz = list(reconstruction.iter_xyz(chunk_points=1000))
    assert len(xyz) == 1
    x, y, z = xyz[0]
    assert x.size == y.size == z.size == 8
    assert np.all(np.isfinite(z))


def test_r20_6_does_not_use_roof_or_canopy_cells_as_reconstruction_anchors():
    mantle = _mantle()
    mantle.reliable[:] = False
    mantle.reliable[2, 1] = True
    mantle.veto_guard.roof_candidate[2, 1] = True

    reconstruction = build_mantle_ground_reconstruction(
        mantle, spacing_m=0.25,
    )
    assert reconstruction.point_count == 0


def test_r20_6_breakline_is_never_synthesised():
    mantle = _mantle()
    mantle.breakline[2, 2] = True
    reconstruction = build_mantle_ground_reconstruction(
        mantle, spacing_m=0.25,
    )
    assert not reconstruction.fill_mask[2, 2]
    assert reconstruction.fill_mask[3, 2]


class _Base:
    def __init__(self, mantle):
        self.params = GroundEngineParams(chunk_size=1234)
        self.mantle = mantle

    def classify_xyz(self, x, y, z):
        return np.full(len(x), 2, dtype=np.uint8)


def test_complete_model_keeps_measured_model_and_exposes_synthetic_channel():
    mantle = _mantle()
    reconstruction = build_mantle_ground_reconstruction(
        mantle, spacing_m=0.25, max_points=1000,
    )
    base = _Base(mantle)
    model = UniversalCompleteGroundModel(
        base, reconstruction, spacing_m=reconstruction.effective_spacing,
    )

    assert model.mantle is mantle
    assert model.params.chunk_size == 1234
    assert model.synthetic_fill_point_count == 8
    assert model.reconstructed_cell_count == 2
    assert model.engine_name == "Universal Ground R20.6"
    assert list(model.iter_synthetic_fill_xyz())


def test_reconstruction_does_not_mutate_preserved_mantle():
    mantle = _mantle()
    before_surface = mantle.surface.copy()
    before_reliable = mantle.reliable.copy()
    before_inferred = mantle.inferred.copy()
    before_hidden = mantle.possible_no_ground_observation.copy()

    build_mantle_ground_reconstruction(mantle, spacing_m=0.25)

    np.testing.assert_array_equal(mantle.surface, before_surface)
    np.testing.assert_array_equal(mantle.reliable, before_reliable)
    np.testing.assert_array_equal(mantle.inferred, before_inferred)
    np.testing.assert_array_equal(
        mantle.possible_no_ground_observation, before_hidden
    )


def test_r20_6_reconstructs_reliable_cells_without_accepted_ground():
    mantle = _mantle()
    measured = np.ones((5, 5), dtype=bool)
    measured[1, 1] = False

    reconstruction = build_mantle_ground_reconstruction(
        mantle,
        measured_ground_cells=measured,
        spacing_m=0.25,
        max_points=1000,
    )

    assert reconstruction.fill_mask[1, 1]
    assert reconstruction.source_state[1, 1] == 3
    assert reconstruction.reliable_missing_cell_count == 1


def test_r20_6_uses_only_accepted_measured_ground_as_projection_anchor():
    mantle = _mantle()
    measured = np.zeros((5, 5), dtype=bool)
    measured[2, 1] = True
    mantle.reliable[:] = False
    mantle.reliable[2, 1] = True
    mantle.reliable[2, 2] = True

    reconstruction = build_mantle_ground_reconstruction(
        mantle,
        measured_ground_cells=measured,
        spacing_m=0.25,
    )
    assert reconstruction.fill_mask[2, 2]


def test_elevated_reliable_cell_is_projected_as_hidden_not_copied():
    mantle = _mantle()
    measured = np.ones((5, 5), dtype=bool)
    measured[2, 3] = False
    mantle.reliable[2, 3] = True
    mantle.observed[2, 3] = True
    mantle.veto_guard.roof_candidate[2, 3] = True
    mantle.surface[2, 3] = 108.0  # visible elevated object
    # Adjacent measured terrain is around 100.x and slopes gently in X.
    before = float(mantle.surface[2, 3])

    reconstruction = build_mantle_ground_reconstruction(
        mantle,
        measured_ground_cells=measured,
        spacing_m=0.25,
        max_points=1000,
    )

    assert reconstruction.fill_mask[2, 3]
    assert reconstruction.source_state[2, 3] == 2
    assert reconstruction.source_state[2, 3] != 3
    assert float(reconstruction.surface[2, 3]) < before - 1.0


def test_elevated_cell_never_becomes_reconstruction_anchor():
    mantle = _mantle()
    measured = np.zeros((5, 5), dtype=bool)
    measured[2, 2] = True
    mantle.reliable[:] = False
    mantle.reliable[2, 2] = True
    mantle.veto_guard.canopy_candidate[2, 2] = True

    reconstruction = build_mantle_ground_reconstruction(
        mantle,
        measured_ground_cells=measured,
        spacing_m=0.25,
    )
    assert reconstruction.point_count == 0
