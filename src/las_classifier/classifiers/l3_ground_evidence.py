from __future__ import annotations

import logging
from dataclasses import dataclass
from time import perf_counter
from typing import Callable

import numpy as np

from ..cloud.model import CloudModel
from ..ground.local_geometry import normal_alignment, point_normals
from ..ground.types import GroundEngineParams
from ..terrain.l3_support import (
    L3SupportGrid,
    L3SupportParams,
    build_l3_support_grid,
)
from ..terrain.schema import SourceInspection, SourceType
from ..terrain.source_inspector import inspect_source
from .adaptive_ptd import (
    AdaptivePTDModel,
    GROUND_CLASS,
    NON_GROUND_CLASS,
    run_adaptive_ptd,
)
from .l3_cloth import (
    L3ClothEvidence,
    L3ClothParams,
    build_l3_cloth,
)
from .smrf import SMRFModel, SMRFParams, run_smrf


LOGGER = logging.getLogger(
    "las_cafiisica.classifiers.l3_ground_evidence"
)
ProgressCallback = Callable[[int, str], None]


def _emit(
    callback: ProgressCallback | None,
    percent: int,
    message: str,
) -> None:
    if callback is not None:
        callback(max(0, min(100, int(percent))), message)


@dataclass(frozen=True, slots=True)
class L3EvidenceParams:
    minimum_ground_score: float = 0.68
    high_confidence_score: float = 0.82
    max_positive_residual: float = 0.55
    max_negative_residual: float = 0.75


def _evidence_params(
    quality: str,
) -> L3EvidenceParams:
    key = quality.strip().lower()
    if key == "fast":
        return L3EvidenceParams(
            minimum_ground_score=0.73,
            high_confidence_score=0.86,
        )
    if key == "high":
        return L3EvidenceParams(
            minimum_ground_score=0.66,
            high_confidence_score=0.80,
        )
    if key == "extreme":
        return L3EvidenceParams(
            minimum_ground_score=0.64,
            high_confidence_score=0.78,
        )
    return L3EvidenceParams()


def _cloth_params(
    quality: str,
) -> L3ClothParams:
    key = quality.strip().lower()
    if key == "fast":
        return L3ClothParams(
            sample_target=1_000_000,
            fine_resolution=0.55,
            coarse_resolution=1.20,
            threshold=0.28,
            iterations=45,
        )
    if key == "high":
        return L3ClothParams(
            sample_target=4_000_000,
            fine_resolution=0.30,
            coarse_resolution=0.80,
            threshold=0.22,
            iterations=90,
        )
    if key == "extreme":
        return L3ClothParams(
            sample_target=7_000_000,
            fine_resolution=0.25,
            coarse_resolution=0.70,
            threshold=0.20,
            iterations=105,
        )
    return L3ClothParams()


def _support_params(
    quality: str,
) -> L3SupportParams:
    key = quality.strip().lower()
    if key == "fast":
        return L3SupportParams(
            sample_target=1_000_000,
            voxel_size=0.70,
            max_plane_distance=0.36,
            min_ptd_score=0.28,
            saturation_count=6,
        )
    if key == "high":
        return L3SupportParams(
            sample_target=5_000_000,
            voxel_size=0.38,
            max_plane_distance=0.30,
            min_ptd_score=0.22,
            saturation_count=10,
        )
    if key == "extreme":
        return L3SupportParams(
            sample_target=8_000_000,
            voxel_size=0.32,
            max_plane_distance=0.28,
            min_ptd_score=0.20,
            saturation_count=12,
        )
    return L3SupportParams()


def _smrf_params(
    params: GroundEngineParams,
) -> SMRFParams:
    return SMRFParams(
        cell=0.50,
        slope=0.18,
        window=14.0,
        threshold=0.30,
        scalar=1.20,
        fill_spacing=0.25,
        chunk_size=params.chunk_size,
        terrain3d_enabled=False,
    )


@dataclass(slots=True)
class L3GroundEvidenceModel:
    params: GroundEngineParams
    evidence_params: L3EvidenceParams
    ptd: AdaptivePTDModel
    cloth: L3ClothEvidence
    smrf: SMRFModel
    support: L3SupportGrid
    source_inspection: SourceInspection
    engine_name: str = "L3 Ground Evidence R18"

    @property
    def synthetic_fill_point_count(self) -> int:
        return 0

    @property
    def effective_fill_spacing(self) -> float:
        return 0.0

    def _return_score(self, points) -> np.ndarray:
        names = set(points.point_format.dimension_names)
        count = len(points)
        score = np.zeros(count, dtype=np.float64)
        if not {
            "return_number",
            "number_of_returns",
        }.issubset(names):
            return score

        rn = np.asarray(points.return_number, dtype=np.int16)
        nr = np.asarray(points.number_of_returns, dtype=np.int16)

        valid = (rn > 0) & (nr > 0)
        only = valid & (rn == 1) & (nr == 1)
        last = valid & (nr > 1) & (rn == nr)
        intermediate = valid & (nr > 2) & (rn > 1) & (rn < nr)
        first = valid & (nr > 1) & (rn == 1)

        score[only] = 0.72
        score[last] = 1.00
        score[intermediate] = 0.28
        score[first] = 0.10
        return score

    def evidence(
        self,
        points,
        x: np.ndarray,
        y: np.ndarray,
        z: np.ndarray,
    ) -> dict[str, np.ndarray]:
        ptd_score, metrics = self.ptd._confidence(
            x,
            y,
            z,
            points=points,
        )
        neighbour_score = self.support.score_xyz(x, y, z)
        cloth_score = self.cloth.confidence_xyz(x, y, z)
        smrf_ground = (
            self.smrf.classify_xyz(x, y, z)
            == GROUND_CLASS
        )
        filter_score = np.clip(
            0.72 * cloth_score
            + 0.28 * smrf_ground.astype(np.float64),
            0.0,
            1.0,
        )
        return_score = self._return_score(points)

        vertical = np.asarray(
            metrics["vertical_residual"],
            dtype=np.float64,
        )
        positive = np.maximum(vertical, 0.0)
        negative = np.maximum(-vertical, 0.0)
        low_score = (
            np.exp(
                -np.square(
                    positive
                    / max(
                        self.evidence_params.max_positive_residual * 0.55,
                        1e-6,
                    )
                )
            )
            * np.exp(
                -np.square(
                    negative
                    / max(
                        self.evidence_params.max_negative_residual * 0.70,
                        1e-6,
                    )
                )
            )
        )

        normals = point_normals(points)
        alignment = normal_alignment(
            normals,
            metrics["normal"],
        )
        if alignment is None:
            normal_score = np.full(
                x.shape[0],
                0.50,
                dtype=np.float64,
            )
            slope_score = np.full(
                x.shape[0],
                0.50,
                dtype=np.float64,
            )
            normal_known = np.zeros(
                x.shape[0],
                dtype=np.bool_,
            )
        else:
            normal_known = np.isfinite(alignment)
            normal_score = np.where(
                normal_known,
                np.clip(alignment, 0.0, 1.0),
                0.50,
            )

            terrain_normal = np.asarray(
                metrics["normal"],
                dtype=np.float64,
            )
            terrain_slope = np.degrees(
                np.arctan2(
                    np.linalg.norm(
                        terrain_normal[:, :2],
                        axis=1,
                    ),
                    np.maximum(
                        np.abs(terrain_normal[:, 2]),
                        1e-9,
                    ),
                )
            )
            point_slope = np.degrees(
                np.arctan2(
                    np.linalg.norm(
                        normals[:, :2],
                        axis=1,
                    ),
                    np.maximum(
                        np.abs(normals[:, 2]),
                        1e-9,
                    ),
                )
            )
            slope_delta = np.abs(
                point_slope - terrain_slope
            )
            slope_score = np.where(
                normal_known,
                np.exp(
                    -np.square(
                        slope_delta / 18.0
                    )
                ),
                0.50,
            )

        score = (
            0.30 * ptd_score
            + 0.20 * neighbour_score
            + 0.16 * low_score
            + 0.14 * return_score
            + 0.10 * normal_score
            + 0.05 * slope_score
            + 0.05 * filter_score
        )

        penalty = np.zeros(
            x.shape[0],
            dtype=np.float64,
        )
        penalty += np.where(
            vertical
            > self.evidence_params.max_positive_residual,
            0.28,
            0.0,
        )
        penalty += np.where(
            vertical
            > self.evidence_params.max_positive_residual * 1.75,
            0.22,
            0.0,
        )
        penalty += np.where(
            -vertical
            > self.evidence_params.max_negative_residual,
            0.18,
            0.0,
        )
        penalty += np.where(
            neighbour_score < 0.08,
            0.16,
            0.0,
        )
        penalty += np.where(
            normal_known & (normal_score < 0.45),
            0.15,
            0.0,
        )

        edge_limit = max(
            float(self.ptd.max_triangle_edge) * np.sqrt(2.0),
            1e-6,
        )
        penalty += np.where(
            metrics["max_edge"] > edge_limit,
            0.20,
            0.0,
        )

        names = set(points.point_format.dimension_names)
        withheld = np.zeros(
            x.shape[0],
            dtype=np.bool_,
        )
        if "withheld" in names:
            withheld = np.asarray(
                points.withheld,
                dtype=np.bool_,
            )
            penalty[withheld] = 1.0

        score = np.clip(
            score - penalty,
            0.0,
            1.0,
        )

        physical_gate = (
            metrics["valid"]
            & (
                (ptd_score >= 0.34)
                | (
                    (neighbour_score >= 0.52)
                    & (cloth_score >= 0.45)
                )
            )
            & (low_score >= 0.18)
            & ~withheld
        )
        ground = (
            score
            >= self.evidence_params.minimum_ground_score
        ) & physical_gate

        return {
            "score": score,
            "ground": ground,
            "ptd_score": ptd_score,
            "neighbour_score": neighbour_score,
            "low_score": low_score,
            "return_score": return_score,
            "normal_score": normal_score,
            "slope_score": slope_score,
            "filter_score": filter_score,
            "cloth_score": cloth_score,
            "smrf_ground": smrf_ground,
            "vertical_residual": vertical,
            "plane_distance": np.asarray(
                metrics["plane_distance"],
                dtype=np.float64,
            ),
        }

    def confidence_points(
        self,
        points,
        x: np.ndarray,
        y: np.ndarray,
        z: np.ndarray,
    ) -> np.ndarray:
        return self.evidence(
            points,
            x,
            y,
            z,
        )["score"]

    def confidence_xyz(
        self,
        x: np.ndarray,
        y: np.ndarray,
        z: np.ndarray,
    ) -> np.ndarray:
        # Export/viewer paths provide real point records through
        # classify_points. XYZ-only confidence is intentionally conservative.
        ptd_score, _ = self.ptd._confidence(x, y, z)
        cloth = self.cloth.confidence_xyz(x, y, z)
        support = self.support.score_xyz(x, y, z)
        return np.clip(
            0.60 * ptd_score
            + 0.25 * support
            + 0.15 * cloth,
            0.0,
            1.0,
        )

    def classify_points(
        self,
        points,
        x: np.ndarray,
        y: np.ndarray,
        z: np.ndarray,
    ) -> np.ndarray:
        e = self.evidence(
            points,
            x,
            y,
            z,
        )
        classes = np.full(
            x.shape[0],
            NON_GROUND_CLASS,
            dtype=np.uint8,
        )
        classes[e["ground"]] = GROUND_CLASS
        return classes

    def classify_xyz(
        self,
        x: np.ndarray,
        y: np.ndarray,
        z: np.ndarray,
    ) -> np.ndarray:
        confidence = self.confidence_xyz(x, y, z)
        classes = np.full(
            x.shape[0],
            NON_GROUND_CLASS,
            dtype=np.uint8,
        )
        classes[
            confidence
            >= self.evidence_params.minimum_ground_score
        ] = GROUND_CLASS
        return classes

    def iter_synthetic_fill_xyz(self):
        if False:
            yield (
                np.empty(0),
                np.empty(0),
                np.empty(0),
            )

    def iter_viewer_synthetic_fill_xyz(self):
        if False:
            yield (
                np.empty(0),
                np.empty(0),
                np.empty(0),
            )


@dataclass(frozen=True, slots=True)
class L3GroundEvidenceResult:
    model: L3GroundEvidenceModel
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
    high_confidence_ground: int
    medium_confidence_ground: int
    recovered_from_non_ground_input: int
    original_class2_accepted: int
    original_class2_rejected: int
    l3_support_voxels: int
    cloth_sampled_returns: int
    mean_confidence: float
    source_type: str
    source_confidence: float
    synthetic_fill_point_count: int = 0
    engine_name: str = "L3 Ground Evidence R18"
    ground_only: bool = True

    @property
    def point_count(self) -> int:
        return self.ground_count + self.non_ground_count


def run_l3_ground_evidence(
    cloud: CloudModel,
    params: GroundEngineParams | None = None,
    progress: ProgressCallback | None = None,
    source_override: SourceType | str | None = None,
) -> L3GroundEvidenceResult:
    params = params or GroundEngineParams()
    started = perf_counter()
    LOGGER.info("GROUND_ENGINE=L3_GROUND_EVIDENCE_R18")
    LOGGER.info("INPUT_CLASSIFICATION_DECISION_WEIGHT=0")

    inspection = inspect_source(
        cloud,
        override=source_override,
    )
    if inspection.source_type is not SourceType.L3_LIDAR:
        raise RuntimeError(
            "L3 Ground Evidence R18 is L3/LiDAR only. "
            f"Detected source: {inspection.source_type.value}. "
            "P1 must use the observability pipeline, not L3 recovery."
        )

    def ptd_progress(
        percent: int,
        message: str,
    ) -> None:
        _emit(
            progress,
            int(percent * 0.32),
            message,
        )

    ptd_result = run_adaptive_ptd(
        cloud,
        params,
        ptd_progress,
        count_full=False,
    )

    def cloth_progress(
        percent: int,
        message: str,
    ) -> None:
        _emit(
            progress,
            32 + int(percent * 0.18),
            message,
        )

    cloth = build_l3_cloth(
        cloud,
        _cloth_params(params.quality),
        cloth_progress,
    )

    def smrf_progress(
        percent: int,
        message: str,
    ) -> None:
        _emit(
            progress,
            50 + int(percent * 0.12),
            message,
        )

    smrf_result = run_smrf(
        cloud,
        _smrf_params(params),
        smrf_progress,
        count_full=False,
    )

    def support_progress(
        percent: int,
        message: str,
    ) -> None:
        _emit(
            progress,
            62 + int(percent * 0.10),
            message,
        )

    support = build_l3_support_grid(
        cloud,
        ptd_result.model,
        _support_params(params.quality),
        support_progress,
    )

    model = L3GroundEvidenceModel(
        params=params,
        evidence_params=_evidence_params(
            params.quality
        ),
        ptd=ptd_result.model,
        cloth=cloth,
        smrf=smrf_result.model,
        support=support,
        source_inspection=inspection,
    )

    total = cloud.point_count
    scales = cloud.las.header.scales
    offsets = cloud.las.header.offsets

    ground_count = 0
    high_count = 0
    medium_count = 0
    recovered_non2 = 0
    class2_accepted = 0
    class2_rejected = 0
    score_sum = 0.0
    score_n = 0

    _emit(
        progress,
        73,
        "L3 evidence: scoring every measured point",
    )

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

        evidence = model.evidence(
            points,
            x,
            y,
            z,
        )
        score = evidence["score"]
        ground = evidence["ground"]
        ground_count += int(
            np.count_nonzero(ground)
        )

        high = (
            ground
            & (
                score
                >= model.evidence_params.high_confidence_score
            )
        )
        medium = ground & ~high
        high_count += int(np.count_nonzero(high))
        medium_count += int(np.count_nonzero(medium))

        names = set(
            points.point_format.dimension_names
        )
        if "classification" in names:
            original = np.asarray(
                points.classification,
                dtype=np.uint8,
            )
            original2 = original == GROUND_CLASS
            recovered_non2 += int(
                np.count_nonzero(
                    ground & ~original2
                )
            )
            class2_accepted += int(
                np.count_nonzero(
                    ground & original2
                )
            )
            class2_rejected += int(
                np.count_nonzero(
                    ~ground & original2
                )
            )

        score_sum += float(np.sum(score))
        score_n += int(score.size)

        _emit(
            progress,
            73
            + int(
                26
                * stop
                / max(1, total)
            ),
            (
                "L3 evidence scoring "
                f"{stop:,}/{total:,}"
            ),
        )

    non_ground = total - ground_count
    mean_confidence = (
        score_sum / score_n
        if score_n
        else 0.0
    )
    elapsed = perf_counter() - started

    LOGGER.info(
        "L3_EVIDENCE ground=%d non_ground=%d "
        "high=%d medium=%d recovered_non2=%d "
        "class2_accepted=%d class2_rejected=%d "
        "support_voxels=%d cloth_returns=%d "
        "mean_score=%.4f elapsed=%.3f",
        ground_count,
        non_ground,
        high_count,
        medium_count,
        recovered_non2,
        class2_accepted,
        class2_rejected,
        support.keys.size,
        cloth.sampled_return_count,
        mean_confidence,
        elapsed,
    )
    _emit(
        progress,
        100,
        "L3 Ground Evidence R18 complete",
    )

    return L3GroundEvidenceResult(
        model=model,
        ground_count=ground_count,
        non_ground_count=non_ground,
        elapsed_seconds=elapsed,
        analysis=ptd_result.analysis,
        seed_count=ptd_result.seed_count,
        candidate_count=ptd_result.candidate_count,
        ptd_iterations=ptd_result.ptd_iterations,
        detected_gap_count=ptd_result.detected_gap_count,
        supported_gap_count=ptd_result.supported_gap_count,
        occluded_gap_count=ptd_result.occluded_gap_count,
        rejected_gap_count=ptd_result.rejected_gap_count,
        high_confidence_ground=high_count,
        medium_confidence_ground=medium_count,
        recovered_from_non_ground_input=recovered_non2,
        original_class2_accepted=class2_accepted,
        original_class2_rejected=class2_rejected,
        l3_support_voxels=int(support.keys.size),
        cloth_sampled_returns=cloth.sampled_return_count,
        mean_confidence=mean_confidence,
        source_type=inspection.source_type.value,
        source_confidence=inspection.confidence,
    )
