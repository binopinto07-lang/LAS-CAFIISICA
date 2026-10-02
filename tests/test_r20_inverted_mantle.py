"""Synthetic tests for experimental R20 inverted measured-ground mantle."""
from __future__ import annotations

import laspy
import numpy as np

from las_classifier.terrain.dense_spatial_evidence import DenseSpatialEvidenceGrid
from las_classifier.terrain.inverted_mantle import build_inverted_ground_mantle
from las_classifier.terrain.mantle_export import (
    MANTLE_STATE,
    STATE_INFERRED_EMPTY,
    STATE_OBSERVED_RELIABLE,
    export_mantle_diagnostic,
    mantle_state_grid,
)


def _synthetic_grid(*, no_anchors: bool = False, step: bool = False):
    grid = DenseSpatialEvidenceGrid.create(
        np.array([0.0, 0.0]), np.array([3.0, 3.0]),
        spacing=0.05, strong_threshold=0.68, max_cells=10_000,
    )
    x, y, z, scores, geometry = [], [], [], [], []
    for iy in range(7):
        for ix in range(7):
            if (ix, iy) == (5, 5):  # physically absent, not Ground by invention
                continue
            base = 10.0 + 0.02 * ix + (3.0 if step and ix >= 4 else 0.0)
            for j in range(4):
                x.append((ix + 0.5) * grid.cell_size + j * 0.002)
                y.append((iy + 0.5) * grid.cell_size + j * 0.002)
                z.append(base)
                anchored = (ix, iy) != (3, 3) and not no_anchors
                scores.append(0.95 if anchored else 0.0)
                geometry.append(anchored)
    z = np.asarray(z)
    score = np.asarray(scores)
    mask = np.asarray(geometry, dtype=np.bool_)
    residual = np.where(mask, 0.02, np.nan)
    plane = np.where(mask, 0.01, np.inf)
    grid.accumulate(
        x=np.asarray(x), y=np.asarray(y), z=z, ptd_score=score,
        vertical_residual=residual, plane_distance=plane,
        valid_mask=np.ones(z.size, dtype=np.bool_),
        geometry_mask=mask,
        original_class=np.ones(z.size, dtype=np.uint8),
    )
    grid.finalize()
    return grid


def test_r20_recovers_observed_surface_even_when_ptd_facet_is_missing():
    grid = _synthetic_grid()
    mantle = build_inverted_ground_mantle(grid)
    x = np.array([3.5 * grid.cell_size])
    y = np.array([3.5 * grid.cell_size])
    assert mantle.reliable[3, 3]
    assert mantle.recovery_mask(x, y, np.array([10.06]))[0]
    assert not mantle.recovery_mask(x, y, np.array([11.06]))[0]
    assert mantle.observed_cell_count > 0
    assert mantle.inferred_cell_count >= 1


def test_r20_no_anchors_means_no_ground_rescue():
    mantle = build_inverted_ground_mantle(_synthetic_grid(no_anchors=True))
    assert mantle.reliable_cell_count == 0
    assert not mantle.recovery_mask(
        np.array([1.225]), np.array([1.225]), np.array([10.06])
    )[0]


def test_r20_discontinuity_not_bridged_as_flat_ground():
    grid = _synthetic_grid(step=True)
    mantle = build_inverted_ground_mantle(grid)
    assert not mantle.reliable[3, 3]
    assert not mantle.recovery_mask(
        np.array([3.5 * grid.cell_size]),
        np.array([3.5 * grid.cell_size]),
        np.array([10.06]),
    )[0]


def test_r20_inferred_cells_are_separate_from_measured_ground(tmp_path):
    mantle = build_inverted_ground_mantle(_synthetic_grid())
    states = mantle_state_grid(mantle)
    assert states[3, 3] == STATE_OBSERVED_RELIABLE
    assert not mantle.observed[5, 5]
    assert states[5, 5] == STATE_INFERRED_EMPTY
    path = export_mantle_diagnostic(mantle, tmp_path / "mantle.las", chunk_size=7)
    with laspy.open(path) as reader:
        las = reader.read()
        assert reader.header.parse_crs().to_epsg() == 3763
        assert MANTLE_STATE in las.point_format.dimension_names
        assert np.any(np.asarray(las[MANTLE_STATE]) == STATE_INFERRED_EMPTY)
        assert not np.any(np.asarray(las.classification) == 2)
        assert np.all(np.asarray(las.synthetic) == 1)
