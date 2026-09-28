from __future__ import annotations

import logging

import numpy as np
from scipy.spatial import Delaunay, QhullError


LOGGER = logging.getLogger("las_cafiisica.ground.tin_delaunay")


def prepare_xy(vertices: np.ndarray):
    data = np.asarray(vertices, dtype=np.float64)
    data = data[:, :3]
    data = data[np.all(np.isfinite(data), axis=1)]
    if data.shape[0] < 3:
        raise ValueError("Terrain TIN requires at least three finite vertices")

    xy = data[:, :2]
    center = (xy.min(axis=0) + xy.max(axis=0)) * 0.5
    scale = float(np.max(np.ptp(xy, axis=0)))
    if scale <= 1e-12:
        raise ValueError("Terrain TIN requires non-zero XY extent")

    norm = (xy - center) / scale
    key = np.rint(norm / 1e-10).astype(np.int64)
    order = np.lexsort((data[:, 2], key[:, 1], key[:, 0]))
    key = key[order]
    keep = np.r_[True, np.any(key[1:] != key[:-1], axis=1)]
    data = data[order[keep]]
    if data.shape[0] < 3:
        raise ValueError("Terrain TIN requires three unique XY vertices")

    xy = data[:, :2]
    center = (xy.min(axis=0) + xy.max(axis=0)) * 0.5
    scale = float(np.max(np.ptp(xy, axis=0)))
    return data, (xy - center) / scale, center, scale


def robust_delaunay(xy: np.ndarray) -> Delaunay:
    failures = []
    for options in ("Qbb Qc Qz Q12", "QJ Qbb Qc Q12"):
        try:
            mesh = Delaunay(xy, qhull_options=options)
            LOGGER.info(
                "TIN_QHULL_OK options=%s points=%d triangles=%d",
                options,
                xy.shape[0],
                mesh.simplices.shape[0],
            )
            return mesh
        except QhullError as exc:
            message = str(exc).splitlines()[0]
            failures.append(message)
            LOGGER.warning(
                "TIN_QHULL_RETRY options=%s error=%s",
                options,
                message,
            )

    centered = xy - np.mean(xy, axis=0)
    singular = np.linalg.svd(
        centered,
        full_matrices=False,
        compute_uv=False,
    )
    if singular.size >= 2 and singular[1] > max(
        1e-12,
        singular[0] * 1e-12,
    ):
        index = np.arange(xy.shape[0], dtype=np.float64) + 1.0
        perturbation = np.column_stack(
            (
                np.sin(index * 1.61803398875),
                np.cos(index * 2.41421356237),
            )
        ) * 1e-9
        try:
            mesh = Delaunay(
                xy + perturbation,
                qhull_options="Qbb Qc Q12",
            )
            LOGGER.warning(
                "TIN_QHULL_RECOVERED points=%d triangles=%d",
                xy.shape[0],
                mesh.simplices.shape[0],
            )
            return mesh
        except QhullError as exc:
            failures.append(str(exc).splitlines()[0])

    raise RuntimeError(
        "Unable to build normalized terrain TIN: "
        + " | ".join(failures)
    )
