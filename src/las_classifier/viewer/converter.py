from __future__ import annotations

import hashlib
import json
import logging
import shutil
import subprocess
from dataclasses import asdict, is_dataclass
from math import ceil
from pathlib import Path
from typing import Callable

import laspy
import numpy as np

from .. import __version__
from ..ground.ground_export import _synthetic_record
from .paths import cache_root, converter_executable


LOGGER = logging.getLogger("las_cafiisica.viewer.converter")
ProgressCallback = Callable[[int, str], None]
SOURCE_CACHE_REVISION = 1
CLASSIFIED_VIEWER_CACHE_REVISION = 3
VIEWER_MAX_GROUND_POINTS = 14_000_000
VIEWER_MAX_NON_GROUND_POINTS = 5_000_000
GROUND_CLASS = np.uint8(2)
NON_GROUND_CLASS = np.uint8(1)


def source_fingerprint(source: str | Path) -> str:
    path = Path(source).expanduser().resolve()
    stat = path.stat()
    payload = (
        f"{path}|{stat.st_size}|{stat.st_mtime_ns}|"
        f"{SOURCE_CACHE_REVISION}"
    ).encode("utf-8", errors="surrogatepass")
    return hashlib.sha256(payload).hexdigest()[:20]


def _params_payload(params) -> object:
    if is_dataclass(params):
        return asdict(params)
    return repr(params)


def classified_fingerprint(
    source: str | Path,
    result,
) -> str:
    payload = {
        "source": source_fingerprint(source),
        "engine": getattr(result, "engine_name", "SMRF"),
        "params": _params_payload(result.model.params),
        "ground_only": bool(getattr(result, "ground_only", False)),
        "version": __version__,
        "viewer_cache_revision": CLASSIFIED_VIEWER_CACHE_REVISION,
        "viewer_ground_cap": VIEWER_MAX_GROUND_POINTS,
        "viewer_non_ground_cap": VIEWER_MAX_NON_GROUND_POINTS,
    }
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()[:20]


def _emit(
    callback: ProgressCallback | None,
    percent: int,
    message: str,
) -> None:
    if callback is not None:
        callback(max(0, min(100, int(percent))), message)


def _scaled(raw, scale: float, offset: float) -> np.ndarray:
    return np.asarray(raw, dtype=np.float64) * float(scale) + float(offset)


def _stride_for(count: int, cap: int) -> int:
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
    ordinal = seen_before + np.arange(indices.size, dtype=np.int64)
    return indices[(ordinal % stride) == 0]


def _classify_points(model, points, x, y, z) -> np.ndarray:
    method = getattr(model, "classify_points", None)
    if method is not None:
        return method(points, x, y, z)
    return model.classify_xyz(x, y, z)


def _write_classified_viewer_laz(
    source: Path,
    output: Path,
    result,
    progress: ProgressCallback | None = None,
    *,
    max_ground_points: int = VIEWER_MAX_GROUND_POINTS,
    max_non_ground_points: int = VIEWER_MAX_NON_GROUND_POINTS,
) -> Path:
    """Create a compact viewer cloud.

    New Ground Engine results are rendered as FINAL GROUND only. Legacy results
    still retain both class 2 and rejected class 1 for comparison.
    """

    model = result.model
    ground_only = bool(getattr(result, "ground_only", False))
    ground_stride = _stride_for(
        int(getattr(result, "ground_count", 0)),
        max_ground_points,
    )
    non_ground_stride = _stride_for(
        int(getattr(result, "non_ground_count", 0)),
        max_non_ground_points,
    )

    output.parent.mkdir(parents=True, exist_ok=True)
    partial = output.with_name(output.name + ".partial")
    if partial.exists():
        partial.unlink()

    LOGGER.info(
        "VIEWER_SAMPLE engine=%s ground_only=%s ground_stride=%d "
        "non_ground_stride=%d synthetic=%d",
        getattr(result, "engine_name", "SMRF"),
        ground_only,
        ground_stride,
        non_ground_stride,
        int(getattr(model, "synthetic_fill_point_count", 0)),
    )

    with laspy.open(source) as reader:
        header = reader.header.copy()
        scales = header.scales
        offsets = header.offsets
        total = int(reader.header.point_count)
        chunk_size = int(getattr(model.params, "chunk_size", 2_000_000))

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
                    points.classification = classes

                    ground_idx = np.flatnonzero(classes == GROUND_CLASS)
                    non_ground_idx = np.flatnonzero(
                        classes == NON_GROUND_CLASS
                    )
                    keep_ground = _sample_class_indices(
                        ground_idx,
                        seen_before=ground_seen,
                        stride=ground_stride,
                    )
                    ground_seen += int(ground_idx.size)

                    if ground_only:
                        keep = keep_ground
                    else:
                        keep_non_ground = _sample_class_indices(
                            non_ground_idx,
                            seen_before=non_ground_seen,
                            stride=non_ground_stride,
                        )
                        non_ground_seen += int(non_ground_idx.size)
                        keep = np.sort(
                            np.concatenate(
                                (keep_ground, keep_non_ground)
                            )
                        )

                    if keep.size:
                        writer.write_points(points[keep])
                        measured_written += int(keep.size)

                    processed += len(points)
                    if total:
                        _emit(
                            progress,
                            int(60 * processed / total),
                            (
                                "Preparing final ground viewer: "
                                f"{processed:,}/{total:,}"
                            ),
                        )

                fill_total = int(
                    getattr(model, "synthetic_fill_point_count", 0)
                )
                fill_written = 0
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
                        fill_written += len(synthetic)
                        if fill_total:
                            _emit(
                                progress,
                                60
                                + int(
                                    5
                                    * fill_written
                                    / max(1, fill_total)
                                ),
                                (
                                    "Viewer reconstructed ground: "
                                    f"{fill_written:,}/{fill_total:,}"
                                ),
                            )

            partial.replace(output)
        except Exception:
            if partial.exists():
                partial.unlink()
            raise

    LOGGER.info(
        "VIEWER_SAMPLE_DONE file=%s measured=%d synthetic=%d",
        output,
        measured_written,
        int(getattr(model, "synthetic_fill_point_count", 0)),
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
    LOGGER.info("POTREE_COMMAND=%s", subprocess.list2cmdline(command))

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
    root = cache_root() / source_fingerprint(source_path) / "original"
    dataset = root / "potree"
    if (dataset / "metadata.json").is_file():
        _emit(progress, 100, "Original viewport loaded from cache")
        return dataset

    root.mkdir(parents=True, exist_ok=True)
    _run_converter(source_path, dataset, progress)
    return dataset


def prepare_classified(
    source: str | Path,
    result,
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
        _emit(progress, 100, "Final ground viewport loaded from cache")
        return dataset

    root.mkdir(parents=True, exist_ok=True)
    viewer_laz = root / "ground_viewer.laz"

    def sample_progress(percent: int, message: str) -> None:
        _emit(progress, int(percent * 0.65), message)

    _write_classified_viewer_laz(
        source_path,
        viewer_laz,
        result,
        sample_progress,
    )

    def converter_progress(percent: int, message: str) -> None:
        _emit(progress, 65 + int(percent * 0.35), message)

    _run_converter(viewer_laz, dataset, converter_progress)
    return dataset
