from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .breaklines import boundary_triangles, triangle_discontinuity
from .terrain_tin import TerrainTIN


SUPPORTED_GAP = np.uint8(1)
EDGE_GAP = np.uint8(2)
LARGE_UNKNOWN_GAP = np.uint8(3)
DISCONTINUITY_GAP = np.uint8(4)


@dataclass(frozen=True, slots=True)
class GapAnalysis:
    kind: np.ndarray
    supported_mask: np.ndarray
    detected_count: int
    supported_count: int
    rejected_count: int


def detect_gaps(
    tin: TerrainTIN,
    *,
    dense_edge: float,
    max_gap_edge: float,
    discontinuity_limit: float = 0.18,
) -> GapAnalysis:
    edge = tin.max_edges
    candidate = edge > dense_edge
    boundary = boundary_triangles(tin)
    discontinuity = triangle_discontinuity(tin)

    kind = np.zeros(tin.triangle_count, dtype=np.uint8)
    kind[candidate & boundary] = EDGE_GAP
    kind[candidate & (edge > max_gap_edge)] = LARGE_UNKNOWN_GAP
    kind[
        candidate
        & (~boundary)
        & (edge <= max_gap_edge)
        & (discontinuity > discontinuity_limit)
    ] = DISCONTINUITY_GAP

    supported = (
        candidate
        & (~boundary)
        & (edge <= max_gap_edge)
        & (discontinuity <= discontinuity_limit)
    )
    kind[supported] = SUPPORTED_GAP

    detected = int(np.count_nonzero(candidate))
    supported_count = int(np.count_nonzero(supported))
    return GapAnalysis(
        kind=kind,
        supported_mask=supported,
        detected_count=detected,
        supported_count=supported_count,
        rejected_count=detected - supported_count,
    )
