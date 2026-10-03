"""R20.3 regression: measured terrace-face cells can be recovered without bridging a breakline."""
from __future__ import annotations

import numpy as np

from las_classifier.terrain.ground_continuity import build_ground_continuity
from las_classifier.terrain.mantle_veto import VETO_NONE
from tests.test_r20_2_ground_continuity import _synthetic_scene, _xyz


def test_breakline_cell_is_recoverable_as_terminal_face():
    grid, mantle, guard = _synthetic_scene()
    mantle.breakline = np.zeros((grid.ny, grid.nx), dtype=np.bool_)
    mantle.breakline[12, 10] = True

    continuity = build_ground_continuity(grid, mantle, guard)

    x, y, z = _xyz(grid, 10, 12)
    assert continuity.breakline[12, 10]
    assert continuity.connected[12, 10]
    assert continuity.recovery_mask(
        x, y, z, veto_codes=np.array([VETO_NONE], dtype=np.uint8)
    )[0]


def test_breakline_cell_cannot_bridge_to_opposite_terrace():
    grid, mantle, guard = _synthetic_scene()
    mantle.breakline = np.zeros((grid.ny, grid.nx), dtype=np.bool_)
    mantle.breakline[12, 10] = True

    continuity = build_ground_continuity(grid, mantle, guard)

    # The terminal breakline cell itself may be recovered, but the graph must
    # not use it as a propagation source for the cell on the opposite side.
    assert continuity.connected[12, 10]
    assert not continuity.source_anchors[12, 10]
