from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import laspy
import numpy as np
from numpy.typing import NDArray


UNKNOWN_CLASS = np.uint8(0)


@dataclass(slots=True)
class CloudModel:
    """In-memory cloud state with immutable-by-convention source classes."""

    path: Path
    las: laspy.LasData
    xyz: NDArray[np.float64]
    original_class: NDArray[np.uint8]
    working_class: NDArray[np.uint8]

    @property
    def point_count(self) -> int:
        return int(self.xyz.shape[0])
