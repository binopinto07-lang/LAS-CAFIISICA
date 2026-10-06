"""R20.5 universal Ground engine.

P1 photogrammetry, L3/LiDAR and UNKNOWN LAS/LAZ sources execute the same
R20.3 geometry pipeline: PTD -> dense evidence -> inverted mantle -> object
veto -> breakline-safe measured-ground continuity.

Sensor identity never selects a different algorithm. Optional LAS dimensions
may contribute evidence only when their measured values are informative.
"""
from __future__ import annotations

from typing import Callable

from ..cloud.model import CloudModel
from ..ground.types import GroundEngineParams
from .l3_ground_lab import L3GroundLabResult, run_l3_ground_lab
from .l3_inverted_ground import _dense_context
from .l3_mantle_veto import _mantle_with_guard
from .l3_ground_continuity import _continuity_builder
from ..terrain.elevated_surface_guard import elevated_island_mask

ProgressCallback = Callable[[int, str], None]

ENGINE_NAME = "Universal Ground R20.5"
REVISION = "R20.5"


def _r205_mantle_with_guard(context, progress):
    """Preserve R20 mantle; extend its *existing* veto before continuity.

    Candidate objects cannot bootstrap continuity, even when an erroneous PTD
    triangle follows the object's top. The same veto applies to all sensors.
    """
    mantle = _mantle_with_guard(context, progress)
    mask = elevated_island_mask(context, mantle)
    mantle.veto_guard.roof_candidate |= mask
    return mantle



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
        mantle_builder=_r205_mantle_with_guard,
        continuity_builder=_continuity_builder,
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
