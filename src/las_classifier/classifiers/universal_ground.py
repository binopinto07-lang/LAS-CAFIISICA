"""R20.5.1 universal Ground engine.

P1 photogrammetry, L3/LiDAR and UNKNOWN LAS/LAZ sources execute the same
R20.3 geometry pipeline: PTD -> dense evidence -> inverted mantle -> object
veto -> breakline-safe measured-ground continuity.

Sensor identity never selects a different algorithm. Optional LAS dimensions
may contribute evidence only when their measured values are informative.
"""
from __future__ import annotations

from dataclasses import replace
from typing import Callable

import numpy as np

from ..cloud.model import CloudModel
from ..ground.types import GroundEngineParams
from .l3_ground_lab import L3GroundLabResult, run_l3_ground_lab
from .l3_inverted_ground import _dense_context
from .l3_mantle_veto import _mantle_with_guard
from .l3_ground_continuity import _continuity_builder
from ..terrain.elevated_surface_guard import build_elevated_surface_guard
from ..terrain.ground_continuity import build_ground_continuity

ProgressCallback = Callable[[int, str], None]

ENGINE_NAME = "Universal Ground R20.5.1"
REVISION = "R20.5.1"


def _r205_continuity_builder(context, mantle, progress):
    """Block elevated islands in continuity WITHOUT changing the R20 mantle.

    The mantle and its original R20.1 veto remain byte-for-byte independent.
    R20.5.1 creates a separate final veto, then builds continuity with a
    temporary combined guard. This prevents an elevated object from becoming a
    propagation bridge while keeping MANTO diagnostics identical to R20.4.
    """
    if mantle is None or mantle.veto_guard is None:
        raise RuntimeError("R20.5.1 requires the preserved guarded R20 mantle")
    if progress is not None:
        progress(56, "R20.5.1: final-object veto + preserved mantle")

    final_veto = build_elevated_surface_guard(context, mantle)
    original_guard = mantle.veto_guard
    combined_guard = replace(
        original_guard,
        roof_candidate=(
            np.asarray(original_guard.roof_candidate, dtype=np.bool_)
            | final_veto.blocked
        ),
    )
    continuity = build_ground_continuity(
        context,
        mantle,
        combined_guard,
    )
    continuity.final_veto = final_veto
    return continuity



def run_universal_ground(
    cloud: CloudModel,
    params: GroundEngineParams | None = None,
    progress: ProgressCallback | None = None,
) -> L3GroundLabResult:
    """Run one geometry-first procedure for every supported point-cloud source.

    P1 is deliberately NOT redirected to a simpler engine.  When no physical
    Ground point exists below vegetation, R20.5 may support an MDT gap later,
    but this classifier never fabricates that point as measured Ground.
    """
    return run_l3_ground_lab(
        cloud,
        params,
        progress,
        source_override=None,
        context_builder=_dense_context,
        mantle_builder=_mantle_with_guard,
        continuity_builder=_r205_continuity_builder,
        engine_name=ENGINE_NAME,
        revision_label=REVISION,
        collect_gate_diagnostics=True,
        require_lidar=False,
    )


def run_universal_ground_r204(
    cloud: CloudModel,
    params: GroundEngineParams | None = None,
    progress: ProgressCallback | None = None,
) -> L3GroundLabResult:
    """Unmodified R20.4 classification geometry for A/B field comparison."""
    return run_l3_ground_lab(
        cloud,
        params,
        progress,
        source_override=None,
        context_builder=_dense_context,
        mantle_builder=_mantle_with_guard,
        continuity_builder=_continuity_builder,
        engine_name="Universal Ground R20.4",
        revision_label="R20.4",
        collect_gate_diagnostics=True,
        require_lidar=False,
    )
