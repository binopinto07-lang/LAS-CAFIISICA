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


LOGGER = logging.getLogger("las_cafiisica.classifiers.hybrid_ground")
ProgressCallback = Callable[[int, str], None]


def _emit(
    callback: ProgressCallback | None,
    percent: int,
    message: str,
) -> None:
    if callback is not None:
        callback(max(0, min(100, int(percent))), message)


@dataclass(slots=True)
class HybridGroundModel:
    params: GroundEngineParams
    ptd: AdaptivePTDModel
    csf: CSFModel
    engine_name: str = "Hybrid"

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

        # PTD is the geometric authority. CSF can rescue borderline terrain,
        # but it cannot turn a geometrically unsupported point into ground.
        score[ptd_score < 0.35] = 0.0
        score[ptd_score >= 0.88] = np.maximum(
            score[ptd_score >= 0.88],
            ptd_score[ptd_score >= 0.88],
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
        classes = np.full(x.shape[0], NON_GROUND_CLASS, dtype=np.uint8)
        classes[score >= self.params.confidence_threshold] = GROUND_CLASS
        return classes

    def classify_points(
        self,
        points,
        x: np.ndarray,
        y: np.ndarray,
        z: np.ndarray,
    ) -> np.ndarray:
        score = self._score(x, y, z, points=points)
        classes = np.full(x.shape[0], NON_GROUND_CLASS, dtype=np.uint8)
        classes[score >= self.params.confidence_threshold] = GROUND_CLASS
        return classes

    def iter_synthetic_fill_xyz(self):
        yield from self.ptd.iter_synthetic_fill_xyz()


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
    engine_name: str = "Hybrid"
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
    LOGGER.info("GROUND_ENGINE=HYBRID")

    def ptd_progress(percent: int, message: str) -> None:
        _emit(progress, int(percent * 0.62), message)

    ptd_result = run_adaptive_ptd(
        cloud,
        params,
        ptd_progress,
        count_full=False,
    )

    def csf_progress(percent: int, message: str) -> None:
        _emit(progress, 62 + int(percent * 0.14), message)

    csf_result = run_csf(
        cloud,
        params,
        csf_progress,
        count_full=False,
    )

    model = HybridGroundModel(
        params=params,
        ptd=ptd_result.model,
        csf=csf_result.model,
    )

    total = cloud.point_count
    ground_count = 0
    confidence_sum = 0.0
    confidence_n = 0
    scales = cloud.las.header.scales
    offsets = cloud.las.header.offsets

    _emit(progress, 77, "Hybrid: validating ground")
    for start in range(0, total, params.chunk_size):
        stop = min(start + params.chunk_size, total)
        points = cloud.las.points[start:stop]
        x = np.asarray(points.X, dtype=np.float64) * scales[0] + offsets[0]
        y = np.asarray(points.Y, dtype=np.float64) * scales[1] + offsets[1]
        z = np.asarray(points.Z, dtype=np.float64) * scales[2] + offsets[2]

        classes = model.classify_points(points, x, y, z)
        ground_count += int(np.count_nonzero(classes == GROUND_CLASS))

        score = model._score(x, y, z, points=points)
        confidence_sum += float(np.sum(score))
        confidence_n += int(score.size)

        _emit(
            progress,
            77 + int(22 * stop / max(1, total)),
            f"Hybrid validate {stop:,}/{total:,}",
        )

    non_ground = total - ground_count
    mean_confidence = (
        confidence_sum / confidence_n
        if confidence_n
        else 0.0
    )
    elapsed = perf_counter() - started

    LOGGER.info("GROUND_REAL=%d", ground_count)
    LOGGER.info("NON_GROUND=%d", non_ground)
    LOGGER.info(
        "GAPS_TOTAL=%d GAPS_SUPPORTED=%d GAPS_OCCLUDED=%d GAPS_REJECTED=%d",
        ptd_result.detected_gap_count,
        ptd_result.supported_gap_count,
        ptd_result.occluded_gap_count,
        ptd_result.rejected_gap_count,
    )
    LOGGER.info(
        "SYNTHETIC_POINTS=%d",
        model.synthetic_fill_point_count,
    )
    LOGGER.info("GROUND_CONFIDENCE_MEAN=%.4f", mean_confidence)
    LOGGER.info("PROCESSING_TIME=%.3f", elapsed)

    _emit(progress, 100, "Hybrid ground complete")
    return HybridGroundResult(
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
        synthetic_fill_point_count=model.synthetic_fill_point_count,
        mean_confidence=mean_confidence,
    )
