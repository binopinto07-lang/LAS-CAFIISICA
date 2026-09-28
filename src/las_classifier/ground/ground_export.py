from __future__ import annotations

import logging
from pathlib import Path
from typing import Callable

import laspy
import numpy as np


LOGGER = logging.getLogger("las_cafiisica.ground.ground_export")
GROUND_CLASS = np.uint8(2)
ProgressCallback = Callable[[int, str], None]


def _scaled(raw, scale: float, offset: float) -> np.ndarray:
    return (
        np.asarray(raw, dtype=np.float64) * float(scale)
        + float(offset)
    )


def _classify_points(model, points, x, y, z) -> np.ndarray:
    method = getattr(model, "classify_points", None)
    if method is not None:
        return method(points, x, y, z)
    return model.classify_xyz(x, y, z)


def _synthetic_record(
    header: laspy.LasHeader,
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
) -> laspy.ScaleAwarePointRecord:
    count = int(x.size)
    points = laspy.ScaleAwarePointRecord.zeros(
        count,
        header=header,
    )
    points.x = x
    points.y = y
    points.z = z
    points.classification = np.full(
        count,
        GROUND_CLASS,
        dtype=np.uint8,
    )

    try:
        points.synthetic = np.ones(count, dtype=np.uint8)
    except Exception:
        LOGGER.warning("SYNTHETIC_FLAG_UNAVAILABLE")

    names = set(header.point_format.dimension_names)
    if "return_number" in names:
        points.return_number = np.ones(count, dtype=np.uint8)
    if "number_of_returns" in names:
        points.number_of_returns = np.ones(count, dtype=np.uint8)

    return points


def export_ground_only(
    source_path: str | Path,
    output_path: str | Path,
    model,
    progress: ProgressCallback | None = None,
    *,
    include_synthetic: bool = True,
) -> Path:
    """Export only measured ground plus optional reconstructed ground.

    The source LAS header is copied as-is so valid source CRS, scale, offset,
    LAS version and point format are preserved. The input file is never
    modified.
    """

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

    with laspy.open(source) as reader:
        header = reader.header.copy()
        scales = header.scales
        offsets = header.offsets
        total = int(reader.header.point_count)
        chunk_size = int(getattr(model.params, "chunk_size", 2_000_000))
        do_compress = output.suffix.lower() == ".laz"

        real_ground = 0
        synthetic_written = 0

        try:
            with laspy.open(
                temporary,
                mode="w",
                header=header,
                do_compress=do_compress,
            ) as writer:
                processed = 0
                for points in reader.chunk_iterator(chunk_size):
                    x = _scaled(points.X, scales[0], offsets[0])
                    y = _scaled(points.Y, scales[1], offsets[1])
                    z = _scaled(points.Z, scales[2], offsets[2])
                    classes = _classify_points(
                        model,
                        points,
                        x,
                        y,
                        z,
                    )
                    keep = classes == GROUND_CLASS
                    if np.any(keep):
                        ground_points = points[keep]
                        ground_points.classification = np.full(
                            int(np.count_nonzero(keep)),
                            GROUND_CLASS,
                            dtype=np.uint8,
                        )
                        writer.write_points(ground_points)
                        real_ground += len(ground_points)

                    processed += len(points)
                    if progress is not None and total:
                        progress(
                            int(82 * processed / total),
                            (
                                "Export ground real: "
                                f"{processed:,}/{total:,}"
                            ),
                        )

                if include_synthetic:
                    fill_total = int(
                        getattr(
                            model,
                            "synthetic_fill_point_count",
                            0,
                        )
                    )
                    iterator = getattr(
                        model,
                        "iter_synthetic_fill_xyz",
                        None,
                    )
                    if iterator is not None:
                        for x, y, z in iterator():
                            synthetic = _synthetic_record(
                                header,
                                np.asarray(x, dtype=np.float64),
                                np.asarray(y, dtype=np.float64),
                                np.asarray(z, dtype=np.float64),
                            )
                            writer.write_points(synthetic)
                            synthetic_written += len(synthetic)
                            if progress is not None and fill_total:
                                progress(
                                    82
                                    + int(
                                        18
                                        * synthetic_written
                                        / max(1, fill_total)
                                    ),
                                    (
                                        "Reconstruct ground: "
                                        f"{synthetic_written:,}/"
                                        f"{fill_total:,}"
                                    ),
                                )

            temporary.replace(output)
        except Exception:
            if temporary.exists():
                temporary.unlink()
            raise

    LOGGER.info("GROUND_ONLY_EXPORT=%s", output)
    LOGGER.info("GROUND_REAL=%d", real_ground)
    LOGGER.info("SYNTHETIC_POINTS=%d", synthetic_written)
    LOGGER.info("GROUND_FINAL=%d", real_ground + synthetic_written)
    if progress is not None:
        progress(100, "Ground-only export complete")
    return output
