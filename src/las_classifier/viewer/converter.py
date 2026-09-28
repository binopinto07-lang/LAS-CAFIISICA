from __future__ import annotations

import hashlib
import json
import logging
import shutil
import subprocess
from dataclasses import asdict
from pathlib import Path
from typing import Callable

from .. import __version__
from ..classifiers.smrf import SMRFResult
from ..cloud.exporter import export_classified
from .paths import cache_root, converter_executable


LOGGER = logging.getLogger("las_cafiisica.viewer.converter")
ProgressCallback = Callable[[int, str], None]
VIEWER_CACHE_REVISION = 1


def source_fingerprint(source: str | Path) -> str:
    path = Path(source).expanduser().resolve()
    stat = path.stat()
    payload = (
        f"{path}|{stat.st_size}|{stat.st_mtime_ns}|"
        f"{VIEWER_CACHE_REVISION}"
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
        "viewer_cache_revision": VIEWER_CACHE_REVISION,
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
    classified_laz = root / "classified_smrf.laz"

    def export_progress(percent: int, message: str) -> None:
        _emit(
            progress,
            int(percent * 0.55),
            "Classified cloud: " + message,
        )

    export_classified(
        source_path,
        classified_laz,
        result.model,
        export_progress,
    )

    def converter_progress(percent: int, message: str) -> None:
        _emit(
            progress,
            55 + int(percent * 0.45),
            message,
        )

    _run_converter(
        classified_laz,
        dataset,
        converter_progress,
    )
    return dataset
