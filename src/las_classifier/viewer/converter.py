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
from pyproj import CRS

from .. import __version__
from ..ground.ground_export import _synthetic_record
from ..terrain.mantle_export import export_mantle_diagnostic
from .paths import cache_root, converter_executable


LOGGER = logging.getLogger("las_cafiisica.viewer.converter")
ProgressCallback = Callable[[int, str], None]
SOURCE_CACHE_REVISION = 3
CLASSIFIED_VIEWER_CACHE_REVISION = 5
VIEWER_MAX_ORIGINAL_POINTS = 25_000_000
VIEWER_MAX_GROUND_POINTS = 14_000_000
VIEWER_MAX_NON_GROUND_POINTS = 5_000_000
VIEWER_SAFE_SCALE = 0.001
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


def _safe_viewer_header(source_header: laspy.LasHeader) -> laspy.LasHeader:
    """Return a normalized header that PotreeConverter 2.1 handles reliably."""

    header = laspy.LasHeader(
        point_format=3,
        version="1.2",
    )
    mins = np.asarray(
        source_header.mins,
        dtype=np.float64,
    )
    header.scales = np.full(
        3,
        VIEWER_SAFE_SCALE,
        dtype=np.float64,
    )
    header.offsets = (
        np.floor(mins / 1000.0) * 1000.0
    ).astype(np.float64)

    # The application works exclusively in ETRS89 / Portugal TM06.
    # Do not trust malformed source CRS VLRs in preview caches (some valid
    # production clouds contain stale GeoKey values such as EPSG:11108).
    header.add_crs(CRS.from_epsg(3763))

    return header


def _copy_standard_viewer_points(
    source_points,
    header: laspy.LasHeader,
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    *,
    classifications: np.ndarray | None = None,
) -> laspy.ScaleAwarePointRecord:
    target = laspy.ScaleAwarePointRecord.zeros(
        len(x),
        header=header,
    )
    target.x = np.asarray(x, dtype=np.float64)
    target.y = np.asarray(y, dtype=np.float64)
    target.z = np.asarray(z, dtype=np.float64)

    source_names = set(
        source_points.point_format.dimension_names
    )
    target_names = set(
        target.point_format.dimension_names
    )
    for name in (
        "intensity",
        "return_number",
        "number_of_returns",
        "scan_direction_flag",
        "edge_of_flight_line",
        "synthetic",
        "key_point",
        "withheld",
        "scan_angle_rank",
        "user_data",
        "point_source_id",
        "gps_time",
        "red",
        "green",
        "blue",
    ):
        if name in source_names and name in target_names:
            try:
                target[name] = np.asarray(
                    source_points[name]
                )
            except Exception:
                LOGGER.debug(
                    "VIEWER_DIMENSION_COPY_SKIPPED name=%s",
                    name,
                    exc_info=True,
                )

    if classifications is not None:
        target.classification = np.asarray(
            classifications,
            dtype=np.uint8,
        )
    elif "classification" in source_names:
        target.classification = np.asarray(
            source_points.classification,
            dtype=np.uint8,
        )

    return target


def _write_original_viewer_laz(
    source: Path,
    output: Path,
    progress: ProgressCallback | None = None,
    *,
    max_points: int = VIEWER_MAX_ORIGINAL_POINTS,
) -> Path:
    """Write a bounded, 1 mm normalized original preview cloud.

    PotreeConverter 2.1 exit code 123 is triggered by strict bounding-box
    validation on some LAS files with high-precision scales/bounds. Rewriting
    only the viewer cache avoids modifying the user's source LAS.
    """

    output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    partial = output.with_name(
        output.name + ".partial"
    )
    if partial.exists():
        partial.unlink()

    with laspy.open(source) as reader:
        total = int(reader.header.point_count)
        stride = _stride_for(
            total,
            max_points,
        )
        header = _safe_viewer_header(
            reader.header
        )
        processed = 0
        written = 0

        try:
            with laspy.open(
                partial,
                mode="w",
                header=header,
                do_compress=True,
            ) as writer:
                for points in reader.chunk_iterator(
                    2_000_000
                ):
                    count = len(points)
                    local = np.arange(count, dtype=np.int64)
                    keep = _spatial_hash_sample_indices(
                        local,
                        points.X,
                        points.Y,
                        points.Z,
                        stride=stride,
                        salt=0x4F524947,
                    )

                    if keep.size:
                        selected = points[keep]
                        x = _scaled(
                            selected.X,
                            reader.header.scales[0],
                            reader.header.offsets[0],
                        )
                        y = _scaled(
                            selected.Y,
                            reader.header.scales[1],
                            reader.header.offsets[1],
                        )
                        z = _scaled(
                            selected.Z,
                            reader.header.scales[2],
                            reader.header.offsets[2],
                        )
                        normalized = (
                            _copy_standard_viewer_points(
                                selected,
                                header,
                                x,
                                y,
                                z,
                            )
                        )
                        writer.write_points(
                            normalized
                        )
                        written += len(
                            normalized
                        )

                    processed += count
                    if total:
                        _emit(
                            progress,
                            int(
                                60
                                * processed
                                / total
                            ),
                            (
                                "Preparing original viewer: "
                                f"{processed:,}/{total:,}"
                            ),
                        )

            partial.replace(output)
        except Exception:
            if partial.exists():
                partial.unlink()
            raise

    LOGGER.info(
        "ORIGINAL_VIEWER_NORMALIZED source=%s output=%s "
        "points=%d stride=%d scale=%.6f",
        source,
        output,
        written,
        stride,
        VIEWER_SAFE_SCALE,
    )
    return output


def _spatial_hash_sample_indices(
    indices: np.ndarray,
    raw_x: np.ndarray,
    raw_y: np.ndarray,
    raw_z: np.ndarray,
    *,
    stride: int,
    salt: int = 0,
) -> np.ndarray:
    """Order-independent viewer decimation.

    Point-index stride sampling produced visible P1 bands/holes because source
    point order is spatially structured. Hashing integer XYZ keeps roughly the
    same cap while breaking that aliasing and is invariant to input order.
    """
    indices = np.asarray(indices, dtype=np.int64)
    if indices.size == 0 or stride <= 1:
        return indices

    x = np.asarray(raw_x, dtype=np.int64)[indices].astype(np.uint64, copy=False)
    y = np.asarray(raw_y, dtype=np.int64)[indices].astype(np.uint64, copy=False)
    z = np.asarray(raw_z, dtype=np.int64)[indices].astype(np.uint64, copy=False)

    h = (
        x * np.uint64(0x9E3779B185EBCA87)
        ^ y * np.uint64(0xC2B2AE3D27D4EB4F)
        ^ z * np.uint64(0x165667B19E3779F9)
        ^ np.uint64(int(salt) & 0xFFFFFFFFFFFFFFFF)
    )
    h ^= h >> np.uint64(30)
    h *= np.uint64(0xBF58476D1CE4E5B9)
    h ^= h >> np.uint64(27)
    h *= np.uint64(0x94D049BB133111EB)
    h ^= h >> np.uint64(31)
    return indices[(h % np.uint64(stride)) == 0]


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
        source_header = reader.header
        header = _safe_viewer_header(
            source_header
        )
        scales = source_header.scales
        offsets = source_header.offsets
        total = int(reader.header.point_count)
        chunk_size = int(getattr(model.params, "chunk_size", 2_000_000))

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
                    keep_ground = _spatial_hash_sample_indices(
                        ground_idx,
                        points.X,
                        points.Y,
                        points.Z,
                        stride=ground_stride,
                        salt=0x47524F554E44,
                    )

                    if ground_only:
                        keep = keep_ground
                    else:
                        keep_non_ground = _spatial_hash_sample_indices(
                            non_ground_idx,
                            points.X,
                            points.Y,
                            points.Z,
                            stride=non_ground_stride,
                            salt=0x4E4F4E47524F554E,
                        )
                        keep = np.sort(
                            np.concatenate(
                                (keep_ground, keep_non_ground)
                            )
                        )

                    if keep.size:
                        selected = points[keep]
                        measured = _copy_standard_viewer_points(
                            selected,
                            header,
                            x[keep],
                            y[keep],
                            z[keep],
                            classifications=classes[keep],
                        )
                        writer.write_points(measured)
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
                    "iter_viewer_synthetic_fill_xyz",
                    None,
                )
                if iterator is None:
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
        cwd=str(converter.parent),
        creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
    )
    assert proc.stdout is not None
    output_tail: list[str] = []
    for line in proc.stdout:
        clean = line.rstrip("\r\n")
        if clean:
            LOGGER.info("POTREE %s", clean)
            output_tail.append(clean)
            if len(output_tail) > 30:
                del output_tail[0]
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
        detail = "\n".join(output_tail[-12:])
        hint = ""
        if code == 123:
            hint = (
                " PotreeConverter 2.1 reportou erro 123, normalmente ligado "
                "a um ponto fora do bounding box declarado; a cloud de viewer "
                "já foi normalizada para escala segura de 1 mm."
            )
        raise RuntimeError(
            f"PotreeConverter terminou com código {code}; "
            f"metadata.json não foi criado em {output}.{hint}"
            + (f"\nÚltimas mensagens Potree:\n{detail}" if detail else "")
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
    viewer_laz = root / "original_viewer.laz"

    def sample_progress(
        percent: int,
        message: str,
    ) -> None:
        _emit(
            progress,
            int(percent * 0.65),
            message,
        )

    _write_original_viewer_laz(
        source_path,
        viewer_laz,
        sample_progress,
    )

    def converter_progress(
        percent: int,
        message: str,
    ) -> None:
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


def prepare_mantle(
    source: str | Path,
    result,
    progress: ProgressCallback | None = None,
) -> Path:
    """R20 diagnostic cloud. Keep it separate from Ground Only + viewer cache."""
    source_path = Path(source).expanduser().resolve()
    mantle = getattr(result.model, "mantle", None)
    if mantle is None:
        raise ValueError("This result has no R20 inverted mantle")

    root = (
        cache_root()
        / source_fingerprint(source_path)
        / "mantle"
        / classified_fingerprint(source_path, result)
    )
    dataset = root / "potree"
    if (dataset / "metadata.json").is_file():
        _emit(progress, 100, "R20 mantle viewer loaded from cache")
        return dataset

    root.mkdir(parents=True, exist_ok=True)
    viewer_laz = root / "mantle_viewer.laz"

    def export_progress(percent: int, message: str) -> None:
        _emit(progress, int(percent * 0.65), message)

    export_mantle_diagnostic(mantle, viewer_laz, export_progress)

    def converter_progress(percent: int, message: str) -> None:
        _emit(progress, 65 + int(percent * 0.35), message)

    _run_converter(viewer_laz, dataset, converter_progress)
    return dataset
