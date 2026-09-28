from __future__ import annotations

import hashlib
import json
import logging
import shutil
import subprocess
from dataclasses import asdict
from math import ceil
from pathlib import Path
from typing import Callable

import laspy
import numpy as np
from pyproj import CRS

from .. import __version__
from ..classifiers.smrf import (
    GROUND_CLASS,
    NON_GROUND_CLASS,
    SMRFResult,
)
from ..cloud.crs import WORKING_EPSG
from ..cloud.exporter import _synthetic_record
from .paths import cache_root, converter_executable


LOGGER = logging.getLogger("las_cafiisica.viewer.converter")
ProgressCallback = Callable[[int, str], None]
SOURCE_CACHE_REVISION = 1
CLASSIFIED_VIEWER_CACHE_REVISION = 2
VIEWER_MAX_GROUND_POINTS = 12_000_000
VIEWER_MAX_NON_GROUND_POINTS = 8_000_000


def source_fingerprint(source: str | Path) -> str:
    path = Path(source).expanduser().resolve()
    stat = path.stat()
    payload = (
        f"{path}|{stat.st_size}|{stat.st_mtime_ns}|"
        f"{SOURCE_CACHE_REVISION}"
    ).encode("utf-8", errors="surrogatepass")
    return hashlib.sha256(payload).hexdigest()[:20]


def classified_fingerprint(
    source: str | Path,
    result: SMRFResult,
) -> str:
    payload = {
        "source": source_fingerprint(source),
        "params": asdict(result.model.params),
        "version": __version__,
        "viewer_cache_revision": CLASSIFIED_VIEWER_CACHE_REVISION,
        "viewer_ground_cap": VIEWER_MAX_GROUND_POINTS,
        "viewer_non_ground_cap": VIEWER_MAX_NON_GROUND_POINTS,
    }
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()[:20]


def _emit(
    callback: ProgressCallback | None,
    percent: int,
    message: str,
) -> None:
    if callback is not None:
        callback(max(0, min(100, int(percent))), message)


def _scaled(
    raw,
    scale: float,
    offset: float,
) -> np.ndarray:
    values = np.asarray(raw, dtype=np.float64)
    values *= float(scale)
    values += float(offset)
    return values


def _stride_for(
    count: int,
    cap: int,
) -> int:
    if count <= 0 or cap <= 0:
        return 1
    return max(1, int(ceil(count / cap)))


def _sample_class_indices(
    indices: np.ndarray,
    *,
    seen_before: int,
    stride: int,
) -> np.ndarray:
    if indices.size == 0 or stride <= 1:
        return indices

    ordinal = (
        seen_before
        + np.arange(indices.size, dtype=np.int64)
    )
    return indices[
        (ordinal % stride) == 0
    ]


def _write_classified_viewer_laz(
    source: Path,
    output: Path,
    result: SMRFResult,
    progress: ProgressCallback | None = None,
    *,
    max_ground_points: int = VIEWER_MAX_GROUND_POINTS,
    max_non_ground_points: int = VIEWER_MAX_NON_GROUND_POINTS,
) -> Path:
    """Create a compact, class-faithful cloud only for 3D display.

    Full-resolution export remains handled by cloud.exporter.export_classified.
    The viewer cloud samples measured ground/non-ground independently so small
    object classes remain visible, while every synthetic terrain-fill point is
    retained.
    """

    model = result.model
    ground_stride = _stride_for(
        result.ground_count,
        max_ground_points,
    )
    non_ground_stride = _stride_for(
        result.non_ground_count,
        max_non_ground_points,
    )

    output.parent.mkdir(parents=True, exist_ok=True)
    partial = output.with_name(output.name + ".partial")
    if partial.exists():
        partial.unlink()

    LOGGER.info(
        "VIEWER_SAMPLE ground_count=%d ground_stride=%d "
        "non_ground_count=%d non_ground_stride=%d synthetic_fill=%d",
        result.ground_count,
        ground_stride,
        result.non_ground_count,
        non_ground_stride,
        model.synthetic_fill_point_count,
    )

    with laspy.open(source) as reader:
        header = reader.header.copy()
        header.add_crs(
            CRS.from_epsg(WORKING_EPSG),
            keep_compatibility=True,
        )
        scales = header.scales
        offsets = header.offsets
        total = int(reader.header.point_count)

        ground_seen = 0
        non_ground_seen = 0
        measured_written = 0

        try:
            with laspy.open(
                partial,
                mode="w",
                header=header,
                do_compress=True,
            ) as writer:
                processed = 0

                for points in reader.chunk_iterator(
                    model.params.chunk_size
                ):
                    x = _scaled(
                        points.X,
                        scales[0],
                        offsets[0],
                    )
                    y = _scaled(
                        points.Y,
                        scales[1],
                        offsets[1],
                    )
                    z = _scaled(
                        points.Z,
                        scales[2],
                        offsets[2],
                    )
                    classes = model.classify_xyz(
                        x,
                        y,
                        z,
                    )
                    points.classification = classes

                    ground_idx = np.flatnonzero(
                        classes == GROUND_CLASS
                    )
                    non_ground_idx = np.flatnonzero(
                        classes == NON_GROUND_CLASS
                    )

                    keep_ground = _sample_class_indices(
                        ground_idx,
                        seen_before=ground_seen,
                        stride=ground_stride,
                    )
                    keep_non_ground = _sample_class_indices(
                        non_ground_idx,
                        seen_before=non_ground_seen,
                        stride=non_ground_stride,
                    )

                    ground_seen += int(ground_idx.size)
                    non_ground_seen += int(non_ground_idx.size)

                    if (
                        keep_ground.size
                        or keep_non_ground.size
                    ):
                        keep = np.sort(
                            np.concatenate(
                                (
                                    keep_ground,
                                    keep_non_ground,
                                )
                            )
                        )
                        writer.write_points(points[keep])
                        measured_written += int(keep.size)

                    processed += len(points)
                    if total:
                        _emit(
                            progress,
                            int(60 * processed / total),
                            (
                                "Viewport classificada: amostrar "
                                f"{processed:,}/{total:,}"
                            ),
                        )

                fill_total = model.synthetic_fill_point_count
                fill_written = 0
                for x, y, z in model.iter_synthetic_fill_xyz():
                    synthetic = _synthetic_record(
                        header,
                        x,
                        y,
                        z,
                    )
                    writer.write_points(synthetic)
                    fill_written += len(synthetic)

                    if fill_total:
                        _emit(
                            progress,
                            60
                            + int(
                                5
                                * fill_written
                                / fill_total
                            ),
                            (
                                "Viewport classificada: preenchimento "
                                f"{fill_written:,}/{fill_total:,}"
                            ),
                        )

            partial.replace(output)
        except Exception:
            if partial.exists():
                partial.unlink()
            raise

    LOGGER.info(
        "VIEWER_SAMPLE_DONE file=%s measured=%d synthetic=%d total=%d",
        output,
        measured_written,
        model.synthetic_fill_point_count,
        measured_written + model.synthetic_fill_point_count,
    )
    return output


def _run_converter(
    source: Path,
    output: Path,
    progress: ProgressCallback | None,
) -> None:
    converter = converter_executable()
    if not converter.is_file():
        raise FileNotFoundError(
            "PotreeConverter não encontrado. Execute o build com a etapa "
            "Preparar Potree + PotreeConverter."
        )

    if output.exists():
        shutil.rmtree(output)
    output.mkdir(parents=True, exist_ok=True)

    command = [
        str(converter),
        str(source),
        "-o",
        str(output),
        "--overwrite",
    ]
    LOGGER.info(
        "POTREE_COMMAND=%s",
        subprocess.list2cmdline(command),
    )

    proc = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
        creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
    )
    assert proc.stdout is not None
    for line in proc.stdout:
        clean = line.rstrip("\r\n")
        if clean:
            LOGGER.info("POTREE %s", clean)
        lower = clean.lower()
        if "counting" in lower:
            _emit(progress, 15, "Potree: counting points")
        elif "indexing" in lower:
            _emit(progress, 40, "Potree: indexing")
        elif "sampling" in lower or "building" in lower:
            _emit(progress, 65, "Potree: building LOD")
        elif "writing" in lower or "flush" in lower:
            _emit(progress, 85, "Potree: writing")

    code = proc.wait()
    metadata = output / "metadata.json"
    if code != 0 or not metadata.is_file():
        raise RuntimeError(
            f"PotreeConverter terminou com código {code}; "
            f"metadata.json não foi criado em {output}"
        )
    _emit(progress, 100, "Viewport 3D ready")


def prepare_original(
    source: str | Path,
    progress: ProgressCallback | None = None,
) -> Path:
    source_path = Path(source).expanduser().resolve()
    root = (
        cache_root()
        / source_fingerprint(source_path)
        / "original"
    )
    dataset = root / "potree"
    if (dataset / "metadata.json").is_file():
        _emit(
            progress,
            100,
            "Original viewport loaded from cache",
        )
        return dataset

    root.mkdir(parents=True, exist_ok=True)
    _run_converter(source_path, dataset, progress)
    return dataset


def prepare_classified(
    source: str | Path,
    result: SMRFResult,
    progress: ProgressCallback | None = None,
) -> Path:
    source_path = Path(source).expanduser().resolve()
    root = (
        cache_root()
        / source_fingerprint(source_path)
        / "classified"
        / classified_fingerprint(source_path, result)
    )
    dataset = root / "potree"
    if (dataset / "metadata.json").is_file():
        _emit(
            progress,
            100,
            "Classified viewport loaded from cache",
        )
        return dataset

    root.mkdir(parents=True, exist_ok=True)
    viewer_laz = root / "classified_viewer.laz"

    def sample_progress(percent: int, message: str) -> None:
        _emit(
            progress,
            int(percent * 0.65),
            message,
        )

    _write_classified_viewer_laz(
        source_path,
        viewer_laz,
        result,
        sample_progress,
    )

    def converter_progress(percent: int, message: str) -> None:
        _emit(
            progress,
            65 + int(percent * 0.35),
            message,
        )

    _run_converter(
        viewer_laz,
        dataset,
        converter_progress,
    )
    return dataset
