from __future__ import annotations

import logging
from dataclasses import dataclass
from time import perf_counter
from typing import Callable

import numpy as np

from ..cloud.model import CloudModel
from ..ground.types import GroundEngineParams
from .adaptive_ptd import (
    AdaptivePTDModel,
    GROUND_CLASS,
    NON_GROUND_CLASS,
    run_adaptive_ptd,
)
from .csf_engine import CSFModel, run_csf
from .terrain3d import (
    Terrain3DParams,
    Terrain3DRefinement,
    build_terrain3d_refinement,
)


LOGGER = logging.getLogger("las_cafiisica.classifiers.hybrid_ground")
ProgressCallback = Callable[[int, str], None]


def _emit(
    callback: ProgressCallback | None,
    percent: int,
    message: str,
) -> None:
    if callback is not None:
        callback(max(0, min(100, int(percent))), message)


def _terrain3d_params(
    params: GroundEngineParams,
) -> Terrain3DParams:
    quality = params.quality
    if quality == "extreme":
        return Terrain3DParams(
            voxel=0.35,
            surface_thickness=0.12,
            min_points=4,
            coherence=0.55,
            seed_ground_fraction=0.40,
            max_normal_angle_deg=88.0,
            target_sample_points=30_000_000,
        )
    if quality == "high":
        return Terrain3DParams(
            voxel=0.40,
            surface_thickness=0.14,
            min_points=4,
            coherence=0.58,
            seed_ground_fraction=0.45,
            max_normal_angle_deg=88.0,
            target_sample_points=20_000_000,
        )
    if quality == "fast":
        return Terrain3DParams(
            voxel=0.75,
            surface_thickness=0.18,
            min_points=5,
            coherence=0.65,
            seed_ground_fraction=0.55,
            max_normal_angle_deg=86.0,
            target_sample_points=6_000_000,
        )
    return Terrain3DParams(
        voxel=0.50,
        surface_thickness=0.16,
        min_points=5,
        coherence=0.62,
        seed_ground_fraction=0.50,
        max_normal_angle_deg=88.0,
        target_sample_points=12_000_000,
    )


@dataclass(slots=True)
class HybridGroundModel:
    params: GroundEngineParams
    ptd: AdaptivePTDModel
    csf: CSFModel
    terrain3d: Terrain3DRefinement | None = None
    engine_name: str = "Hybrid 2.5D + 3D"

    @property
    def synthetic_fill_point_count(self) -> int:
        return self.ptd.synthetic_fill_point_count

    @property
    def effective_fill_spacing(self) -> float:
        return self.ptd.effective_fill_spacing

    def _score(
        self,
        x: np.ndarray,
        y: np.ndarray,
        z: np.ndarray,
        points=None,
    ) -> np.ndarray:
        ptd_score, _ = self.ptd._confidence(
            x,
            y,
            z,
            points=points,
        )
        csf_score = self.csf.confidence_xyz(x, y, z)

        score = 0.84 * ptd_score + 0.16 * csf_score

        # The 2.5D PTD remains the authority for ordinary terrain. It cannot,
        # however, represent vertical or near-vertical vineyard/talude faces
        # because a Delaunay TIN over XY has only one Z for each XY location.
        unsupported_2d = ptd_score < 0.35
        score[unsupported_2d] = 0.0
        strong_ptd = ptd_score >= 0.88
        score[strong_ptd] = np.maximum(
            score[strong_ptd],
            ptd_score[strong_ptd],
        )

        # Hybrid V2 structural rescue: grow a true 3-D surface through voxels
        # with coherent normals that are connected to reliable PTD ground.
        # This is the path that keeps retaining walls, terrace faces and steep
        # taludes without relaxing the vegetation thresholds globally.
        if self.terrain3d is not None:
            terrain3d_ground = self.terrain3d.ground_mask(
                x,
                y,
                z,
            )
            score[terrain3d_ground] = np.maximum(
                score[terrain3d_ground],
                0.99,
            )

        return np.clip(score, 0.0, 1.0)

    def confidence_xyz(
        self,
        x: np.ndarray,
        y: np.ndarray,
        z: np.ndarray,
    ) -> np.ndarray:
        return self._score(x, y, z)

    def confidence_points(
        self,
        points,
        x: np.ndarray,
        y: np.ndarray,
        z: np.ndarray,
    ) -> np.ndarray:
        return self._score(
            x,
            y,
            z,
            points=points,
        )

    def classify_xyz(
        self,
        x: np.ndarray,
        y: np.ndarray,
        z: np.ndarray,
    ) -> np.ndarray:
        score = self._score(x, y, z)
        classes = np.full(
            x.shape[0],
            NON_GROUND_CLASS,
            dtype=np.uint8,
        )
        classes[
            score >= self.params.confidence_threshold
        ] = GROUND_CLASS
        return classes

    def classify_points(
        self,
        points,
        x: np.ndarray,
        y: np.ndarray,
        z: np.ndarray,
    ) -> np.ndarray:
        score = self._score(
            x,
            y,
            z,
            points=points,
        )
        classes = np.full(
            x.shape[0],
            NON_GROUND_CLASS,
            dtype=np.uint8,
        )
        classes[
            score >= self.params.confidence_threshold
        ] = GROUND_CLASS
        return classes

    def iter_synthetic_fill_xyz(self):
        yield from self.ptd.iter_synthetic_fill_xyz()

    def iter_viewer_synthetic_fill_xyz(self):
        yield from self.ptd.iter_viewer_synthetic_fill_xyz()


@dataclass(frozen=True, slots=True)
class HybridGroundResult:
    model: HybridGroundModel
    ground_count: int
    non_ground_count: int
    elapsed_seconds: float
    analysis: object
    seed_count: int
    candidate_count: int
    ptd_iterations: int
    detected_gap_count: int
    supported_gap_count: int
    occluded_gap_count: int
    rejected_gap_count: int
    synthetic_fill_point_count: int
    mean_confidence: float
    terrain3d_voxel_count: int = 0
    terrain3d_seed_voxel_count: int = 0
    engine_name: str = "Hybrid 2.5D + 3D"
    ground_only: bool = True

    @property
    def point_count(self) -> int:
        return self.ground_count + self.non_ground_count


def run_hybrid_ground(
    cloud: CloudModel,
    params: GroundEngineParams | None = None,
    progress: ProgressCallback | None = None,
) -> HybridGroundResult:
    params = params or GroundEngineParams()
    started = perf_counter()
    LOGGER.info("GROUND_ENGINE_START")
    LOGGER.info("GROUND_ENGINE=HYBRID_2_5D_PLUS_3D")

    def ptd_progress(percent: int, message: str) -> None:
        _emit(
            progress,
            int(percent * 0.52),
            message,
        )

    ptd_result = run_adaptive_ptd(
        cloud,
        params,
        ptd_progress,
        count_full=False,
    )

    def csf_progress(percent: int, message: str) -> None:
        _emit(
            progress,
            52 + int(percent * 0.10),
            message,
        )

    csf_result = run_csf(
        cloud,
        params,
        csf_progress,
        count_full=False,
    )

    _emit(
        progress,
        63,
        "Hybrid: building true 3D terrain surface",
    )
    terrain3d = build_terrain3d_refinement(
        cloud,
        ptd_result.model,
        progress,
        _terrain3d_params(params),
    )

    model = HybridGroundModel(
        params=params,
        ptd=ptd_result.model,
        csf=csf_result.model,
        terrain3d=terrain3d,
    )

    total = cloud.point_count
    ground_count = 0
    confidence_sum = 0.0
    confidence_n = 0
    scales = cloud.las.header.scales
    offsets = cloud.las.header.offsets

    _emit(
        progress,
        87 if terrain3d is not None else 77,
        "Hybrid 2.5D + 3D: validating ground",
    )
    validation_start = 87 if terrain3d is not None else 77
    validation_span = 12 if terrain3d is not None else 22

    for start in range(
        0,
        total,
        params.chunk_size,
    ):
        stop = min(
            start + params.chunk_size,
            total,
        )
        points = cloud.las.points[start:stop]
        x = (
            np.asarray(
                points.X,
                dtype=np.float64,
            )
            * scales[0]
            + offsets[0]
        )
        y = (
            np.asarray(
                points.Y,
                dtype=np.float64,
            )
            * scales[1]
            + offsets[1]
        )
        z = (
            np.asarray(
                points.Z,
                dtype=np.float64,
            )
            * scales[2]
            + offsets[2]
        )

        classes = model.classify_points(
            points,
            x,
            y,
            z,
        )
        ground_count += int(
            np.count_nonzero(
                classes == GROUND_CLASS
            )
        )

        score = model._score(
            x,
            y,
            z,
            points=points,
        )
        confidence_sum += float(
            np.sum(score)
        )
        confidence_n += int(score.size)

        _emit(
            progress,
            validation_start
            + int(
                validation_span
                * stop
                / max(1, total)
            ),
            (
                "Hybrid 2.5D + 3D validate "
                f"{stop:,}/{total:,}"
            ),
        )

    non_ground = total - ground_count
    mean_confidence = (
        confidence_sum / confidence_n
        if confidence_n
        else 0.0
    )
    elapsed = perf_counter() - started

    terrain3d_voxels = (
        terrain3d.terrain_voxel_count
        if terrain3d is not None
        else 0
    )
    terrain3d_seeds = (
        terrain3d.seed_voxels
        if terrain3d is not None
        else 0
    )

    LOGGER.info("GROUND_REAL=%d", ground_count)
    LOGGER.info("NON_GROUND=%d", non_ground)
    LOGGER.info(
        "TERRAIN3D_VOXELS=%d TERRAIN3D_SEEDS=%d",
        terrain3d_voxels,
        terrain3d_seeds,
    )
    LOGGER.info(
        "GAPS_TOTAL=%d GAPS_SUPPORTED=%d "
        "GAPS_OCCLUDED=%d GAPS_REJECTED=%d",
        ptd_result.detected_gap_count,
        ptd_result.supported_gap_count,
        ptd_result.occluded_gap_count,
        ptd_result.rejected_gap_count,
    )
    LOGGER.info(
        "SYNTHETIC_POINTS=%d",
        model.synthetic_fill_point_count,
    )
    LOGGER.info(
        "GROUND_CONFIDENCE_MEAN=%.4f",
        mean_confidence,
    )
    LOGGER.info(
        "PROCESSING_TIME=%.3f",
        elapsed,
    )

    _emit(
        progress,
        100,
        "Hybrid 2.5D + 3D ground complete",
    )
    return HybridGroundResult(
        model=model,
        ground_count=ground_count,
        non_ground_count=non_ground,
        elapsed_seconds=elapsed,
        analysis=ptd_result.analysis,
        seed_count=ptd_result.seed_count,
        candidate_count=ptd_result.candidate_count,
        ptd_iterations=ptd_result.ptd_iterations,
        detected_gap_count=(
            ptd_result.detected_gap_count
        ),
        supported_gap_count=(
            ptd_result.supported_gap_count
        ),
        occluded_gap_count=(
            ptd_result.occluded_gap_count
        ),
        rejected_gap_count=(
            ptd_result.rejected_gap_count
        ),
        synthetic_fill_point_count=(
            model.synthetic_fill_point_count
        ),
        mean_confidence=mean_confidence,
        terrain3d_voxel_count=terrain3d_voxels,
        terrain3d_seed_voxel_count=terrain3d_seeds,
    )
