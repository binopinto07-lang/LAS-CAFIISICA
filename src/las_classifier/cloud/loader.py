from __future__ import annotations

import logging
from pathlib import Path

import laspy
import numpy as np

from .model import CloudModel, UNKNOWN_CLASS


LOGGER = logging.getLogger("las_cafiisica.cloud.loader")
IGNORE_INPUT_CLASSIFICATION = True
SUPPORTED_EXTENSIONS = {".las", ".laz"}


def load_cloud(path: str | Path) -> CloudModel:
    """Read LAS/LAZ while preserving input classification for diagnostics only."""

    source = Path(path).expanduser().resolve()
    if source.suffix.lower() not in SUPPORTED_EXTENSIONS:
        raise ValueError(
            f"Unsupported point-cloud format: {source.suffix or '<none>'}"
        )
    if not source.is_file():
        raise FileNotFoundError(source)

    LOGGER.info("INPUT_FILE=%s", source)
    las = laspy.read(source)
    xyz = np.column_stack((las.x, las.y, las.z)).astype(np.float64, copy=False)

    if "classification" in las.point_format.dimension_names:
        original_class = np.asarray(las.classification, dtype=np.uint8).copy()
    else:
        original_class = np.zeros(len(las.points), dtype=np.uint8)

    # Critical invariant: source classification is never adopted as working state.
    working_class = np.full(len(las.points), UNKNOWN_CLASS, dtype=np.uint8)

    LOGGER.info("POINT_COUNT=%d", len(las.points))
    LOGGER.info("IGNORE_INPUT_CLASSIFICATION=%s", IGNORE_INPUT_CLASSIFICATION)

    return CloudModel(
        path=source,
        las=las,
        xyz=xyz,
        original_class=original_class,
        working_class=working_class,
    )
