"""Universal Ground R20.7 — LOW-SEED TIN + progressive measured Ground + reconstruction.

All P1/L3/UNKNOWN sources use the same measured classification pipeline. R20.7
then adds a SEPARATE synthetic Ground layer only for mantle cells explicitly
marked NO_GROUND_OBSERVATION. The measured cloud and preserved R20 mantle are
never rewritten.
"""
from __future__ import annotations

from dataclasses import replace
from typing import Callable

import numpy as np

from ..cloud.model import CloudModel
from ..ground.types import GroundEngineParams
from ..terrain.elevated_surface_guard import (
    ElevatedSurfaceGuard,
    build_elevated_surface_guard,
)
from ..terrain.seed_tin_growth import build_seed_tin_ground_guard
from ..terrain.ground_continuity import build_ground_continuity
from ..terrain.mantle_reconstruction import (
    MantleGroundReconstruction,
    build_mantle_ground_reconstruction,
)
from .l3_ground_continuity import _continuity_builder
from .l3_ground_lab import L3GroundLabResult, run_l3_ground_lab
from .l3_inverted_ground import _dense_context
from .l3_mantle_veto import _mantle_with_guard

ProgressCallback = Callable[[int, str], None]

ENGINE_NAME = "Universal Ground R20.7"
REVISION = "R20.7"


def _r2062_continuity_builder(context, mantle, progress):
    """Preserve the R20.4 mantle while blocking elevated islands in FINAL Ground."""
    if mantle is None or mantle.veto_guard is None:
        raise RuntimeError("R20.5.1+ requires the preserved guarded R20 mantle")
    if progress is not None:
        progress(56, "R20.6.2: object/vertical veto + preserved mantle")

    final_veto = build_elevated_surface_guard(context, mantle)
    original_guard = mantle.veto_guard
    combined_guard = replace(
        original_guard,
        roof_candidate=(
            np.asarray(original_guard.roof_candidate, dtype=np.bool_)
            | final_veto.blocked
        ),
    )
    continuity = build_ground_continuity(context, mantle, combined_guard)
    continuity.final_veto = final_veto
    return continuity


def _r207_continuity_builder(context, mantle, progress):
    """R20.7 low-seed TIN authority + R20.6.2 object veto.

    Both guards are independent from the preserved R20 mantle. Their union is
    applied before continuity and again at FINAL-GROUND point level.
    """
    if mantle is None or mantle.veto_guard is None:
        raise RuntimeError("R20.7 requires the preserved guarded R20 mantle")
    if progress is not None:
        progress(55, "R20.7: low-seed grid + terrain TIN")

    object_guard = build_elevated_surface_guard(context, mantle)
    tin_guard = build_seed_tin_ground_guard(context, mantle)
    blocked = (
        np.asarray(object_guard.blocked, dtype=np.bool_)
        | np.asarray(tin_guard.blocked, dtype=np.bool_)
    )
    final_veto = ElevatedSurfaceGuard(
        origin=np.asarray(mantle.origin, dtype=np.float64).copy(),
        cell_size=float(context.cell_size),
        nx=int(context.nx),
        ny=int(context.ny),
        blocked=blocked,
    )

    original_guard = mantle.veto_guard
    combined_guard = replace(
        original_guard,
        roof_candidate=(
            np.asarray(original_guard.roof_candidate, dtype=np.bool_)
            | blocked
        ),
    )
    continuity = build_ground_continuity(context, mantle, combined_guard)
    continuity.final_veto = final_veto
    continuity.terrain_guard = tin_guard
    if progress is not None:
        progress(
            58,
            "R20.7: "
            f"{tin_guard.seed_count:,} low seeds · "
            f"{tin_guard.blocked_cell_count:,} TIN high cells blocked",
        )
    return continuity


class UniversalCompleteGroundModel:
    """Delegate measured classification and expose reconstructed Ground separately."""

    def __init__(
        self,
        base_model,
        reconstruction: MantleGroundReconstruction,
        *,
        spacing_m: float,
        measured_ground_points: int = 0,
    ) -> None:
        self.base_model = base_model
        self.reconstruction = reconstruction
        self.engine_name = ENGINE_NAME
        self.measured_ground_point_count = int(measured_ground_points)
        self.params = replace(
            base_model.params,
            synthetic_spacing=float(spacing_m),
        )

    def __getattr__(self, name):
        return getattr(self.base_model, name)

    @property
    def synthetic_fill_point_count(self) -> int:
        return int(self.reconstruction.point_count)

    @property
    def effective_fill_spacing(self) -> float:
        return float(self.reconstruction.effective_spacing)

    @property
    def reconstructed_cell_count(self) -> int:
        return int(self.reconstruction.fill_cell_count)

    @property
    def reconstructed_inferred_cell_count(self) -> int:
        return int(self.reconstruction.inferred_cell_count)

    @property
    def reconstructed_hidden_cell_count(self) -> int:
        return int(self.reconstruction.hidden_ground_cell_count)

    @property
    def reconstructed_reliable_missing_cell_count(self) -> int:
        return int(self.reconstruction.reliable_missing_cell_count)

    def iter_synthetic_fill_xyz(self, chunk_points: int = 500_000):
        yield from self.reconstruction.iter_xyz(chunk_points=chunk_points)

    def iter_viewer_synthetic_fill_xyz(self, chunk_points: int = 500_000):
        yield from self.reconstruction.iter_xyz(chunk_points=chunk_points)


def _run_measured(
    cloud: CloudModel,
    params: GroundEngineParams | None,
    progress: ProgressCallback | None,
    *,
    engine_name: str,
    revision_label: str,
    continuity_builder=_r207_continuity_builder,
) -> L3GroundLabResult:
    return run_l3_ground_lab(
        cloud,
        params,
        progress,
        source_override=None,
        context_builder=_dense_context,
        mantle_builder=_mantle_with_guard,
        continuity_builder=continuity_builder,
        engine_name=engine_name,
        revision_label=revision_label,
        collect_gate_diagnostics=True,
        require_lidar=False,
    )


def run_universal_ground(
    cloud: CloudModel,
    params: GroundEngineParams | None = None,
    progress: ProgressCallback | None = None,
) -> L3GroundLabResult:
    """R20.7 FINAL GROUND = measured Ground + reconstructed NO_GROUND_OBSERVATION.

    Reconstructed XYZ are never presented as measured observations. They are
    emitted through the existing synthetic Ground channel (GroundSource=2).
    """
    requested = params or GroundEngineParams()
    result = _run_measured(
        cloud,
        requested,
        progress,
        engine_name=ENGINE_NAME,
        revision_label=REVISION,
    )
    spacing = (
        float(requested.synthetic_spacing)
        if float(requested.synthetic_spacing) > 0.0
        else 0.25
    )
    final_veto = getattr(
        getattr(result.model, "continuity", None),
        "final_veto",
        None,
    )
    reconstruction = build_mantle_ground_reconstruction(
        result.model.mantle,
        measured_ground_cells=result.model.measured_ground_cells,
        elevated_mask=(
            getattr(final_veto, "blocked", None)
            if final_veto is not None else None
        ),
        spacing_m=spacing,
    )
    model = UniversalCompleteGroundModel(
        result.model,
        reconstruction,
        spacing_m=reconstruction.effective_spacing,
        measured_ground_points=result.ground_count,
    )
    if progress is not None:
        progress(
            100,
            "R20.7 FINAL GROUND: "
            f"{result.ground_count:,} measured + "
            f"{model.synthetic_fill_point_count:,} reconstructed",
        )
    return replace(
        result,
        model=model,
        synthetic_fill_point_count=model.synthetic_fill_point_count,
        engine_name=ENGINE_NAME,
    )


def run_universal_ground_r2062(
    cloud: CloudModel,
    params: GroundEngineParams | None = None,
    progress: ProgressCallback | None = None,
) -> L3GroundLabResult:
    """R20.6.2 comparator without the new low-seed TIN guard."""
    requested = params or GroundEngineParams()
    result = _run_measured(
        cloud,
        requested,
        progress,
        engine_name="Universal Ground R20.6.2",
        revision_label="R20.6.2",
        continuity_builder=_r2062_continuity_builder,
    )
    spacing = (
        float(requested.synthetic_spacing)
        if float(requested.synthetic_spacing) > 0.0 else 0.25
    )
    final_veto = getattr(
        getattr(result.model, "continuity", None), "final_veto", None
    )
    reconstruction = build_mantle_ground_reconstruction(
        result.model.mantle,
        measured_ground_cells=result.model.measured_ground_cells,
        elevated_mask=(
            getattr(final_veto, "blocked", None)
            if final_veto is not None else None
        ),
        spacing_m=spacing,
    )
    model = UniversalCompleteGroundModel(
        result.model,
        reconstruction,
        spacing_m=reconstruction.effective_spacing,
        measured_ground_points=result.ground_count,
    )
    model.engine_name = "Universal Ground R20.6.2"
    return replace(
        result,
        model=model,
        synthetic_fill_point_count=model.synthetic_fill_point_count,
        engine_name="Universal Ground R20.6.2",
    )


def run_universal_ground_r2051(
    cloud: CloudModel,
    params: GroundEngineParams | None = None,
    progress: ProgressCallback | None = None,
) -> L3GroundLabResult:
    """R20.5.1 comparator: measured Ground only, preserved mantle."""
    return _run_measured(
        cloud,
        params,
        progress,
        engine_name="Universal Ground R20.5.1",
        revision_label="R20.5.1",
        continuity_builder=_r2062_continuity_builder,
    )


def run_universal_ground_r204(
    cloud: CloudModel,
    params: GroundEngineParams | None = None,
    progress: ProgressCallback | None = None,
) -> L3GroundLabResult:
    """Unmodified R20.4 classification geometry for A/B comparison."""
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
