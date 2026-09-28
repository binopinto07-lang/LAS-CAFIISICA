from __future__ import annotations

import logging
from pathlib import Path
from typing import Callable

import laspy
import numpy as np
from pyproj import CRS

from ..classifiers.smrf import SMRFModel
from .crs import WORKING_EPSG


LOGGER = logging.getLogger("las_cafiisica.cloud.exporter")
ProgressCallback = Callable[[int, str], None]


def _scaled(
    raw,
    scale: float,
    offset: float,
) -> np.ndarray:
    values = np.asarray(raw, dtype=np.float64)
    values *= float(scale)
    values += float(offset)
    return values


def export_classified(
    source_path: str | Path,
    output_path: str | Path,
    model: SMRFModel,
    progress: ProgressCallback | None = None,
) -> Path:
    source = Path(source_path).expanduser().resolve()
    output = Path(output_path).expanduser().resolve()
    if source == output:
        raise ValueError("Output must be different from the source file")
    if output.suffix.lower() not in {".las", ".laz"}:
        raise ValueError("Output must use .las or .laz")

    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(output.name + ".partial")
    if temporary.exists():
        temporary.unlink()

    try:
        with laspy.open(source) as reader:
            header = reader.header.copy()
            header.add_crs(
                CRS.from_epsg(WORKING_EPSG),
                keep_compatibility=True,
            )
            total = int(reader.header.point_count)
            scales = header.scales
            offsets = header.offsets
            do_compress = output.suffix.lower() == ".laz"

            with laspy.open(
                temporary,
                mode="w",
                header=header,
                do_compress=do_compress,
            ) as writer:
                processed = 0
                for points in reader.chunk_iterator(
                    model.params.chunk_size
                ):
                    x = _scaled(points.X, scales[0], offsets[0])
                    y = _scaled(points.Y, scales[1], offsets[1])
                    z = _scaled(points.Z, scales[2], offsets[2])
                    points.classification = model.classify_xyz(
                        x,
                        y,
                        z,
                    )
                    writer.write_points(points)
                    processed += len(points)

                    if progress is not None and total:
                        progress(
                            int(100 * processed / total),
                            (
                                "Export classified: "
                                f"{processed:,}/{total:,}"
                            ),
                        )

        temporary.replace(output)
    except Exception:
        if temporary.exists():
            temporary.unlink()
        raise

    LOGGER.info(
        "CLASSIFIED_EXPORT=%s WORKING_CRS=EPSG:%d",
        output,
        WORKING_EPSG,
    )
    return output
