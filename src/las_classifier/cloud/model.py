from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import laspy
import numpy as np
from numpy.typing import NDArray


UNKNOWN_CLASS = np.uint8(0)


@dataclass(slots=True)
class CloudModel:
    """In-memory cloud state.

    ``original_class`` is an immutable-by-convention copy of the LAS input
    classification. ``working_class`` starts UNKNOWN and deliberately does not
    inherit input classification, so future classifiers cannot accidentally use
    the source classes as their initial answer.
    """

    path: Path
    las: laspy.LasData
    xyz: NDArray[np.float64]
    original_class: NDArray[np.uint8]
    working_class: NDArray[np.uint8]

    @property
    def point_count(self) -> int:
        return int(self.xyz.shape[0])
