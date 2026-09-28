from __future__ import annotations

from math import ceil

import numpy as np

from .terrain_tin import TerrainTIN


def estimate_fill_count(
    tin: TerrainTIN,
    supported_mask: np.ndarray,
    spacing: float,
) -> int:
    if spacing <= 0:
        return 0
    areas = tin.areas[supported_mask]
    if areas.size == 0:
        return 0
    return int(np.sum(np.ceil(areas / (spacing * spacing))))


def iter_triangle_fill(
    tin: TerrainTIN,
    supported_mask: np.ndarray,
    spacing: float,
    *,
    max_points: int = 8_000_000,
    chunk_points: int = 500_000,
):
    triangles = tin.vertices[tin.simplices[supported_mask]]
    if triangles.size == 0:
        return

    requested = max(spacing, 0.02)
    estimated = max(
        1,
        estimate_fill_count(tin, supported_mask, requested),
    )
    if estimated > max_points:
        requested *= np.sqrt(estimated / max_points)

    buffer: list[np.ndarray] = []
    buffered = 0

    for tri in triangles:
        e01 = np.linalg.norm(tri[1] - tri[0])
        e02 = np.linalg.norm(tri[2] - tri[0])
        e12 = np.linalg.norm(tri[2] - tri[1])
        divisions = max(
            2,
            int(ceil(max(e01, e02, e12) / requested)),
        )

        points: list[np.ndarray] = []
        for i in range(1, divisions):
            for j in range(1, divisions - i):
                a = i / divisions
                b = j / divisions
                c = 1.0 - a - b
                if c <= 0.0:
                    continue
                points.append(
                    a * tri[0] + b * tri[1] + c * tri[2]
                )

        if not points:
            continue
        array = np.asarray(points, dtype=np.float64)
        buffer.append(array)
        buffered += array.shape[0]

        if buffered >= chunk_points:
            merged = np.concatenate(buffer, axis=0)
            yield (
                merged[:, 0],
                merged[:, 1],
                merged[:, 2],
            )
            buffer = []
            buffered = 0

    if buffer:
        merged = np.concatenate(buffer, axis=0)
        yield (
            merged[:, 0],
            merged[:, 1],
            merged[:, 2],
        )
