from __future__ import annotations

import os
import sys
from pathlib import Path


def app_root() -> Path:
    if getattr(sys, "frozen", False):
        return Path(sys.executable).resolve().parent
    return Path(__file__).resolve().parents[3]


def viewer_root() -> Path:
    return app_root() / "viewer"


def vendor_root() -> Path:
    return app_root() / "vendor"


def potree_root() -> Path:
    return vendor_root() / "potree"


def converter_executable() -> Path:
    explicit = os.environ.get("LAS_CAFIISICA_POTREE_CONVERTER")
    if explicit:
        return Path(explicit).expanduser().resolve()

    root = vendor_root() / "potreeconverter"
    candidates = [
        root / "PotreeConverter.exe",
        root / "PotreeConverter",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


def cache_root() -> Path:
    local = os.environ.get("LOCALAPPDATA")
    base = Path(local) if local else Path.home() / ".las_cafiisica"
    path = base / "LAS_CAFIISICA" / "viewer_cache"
    path.mkdir(parents=True, exist_ok=True)
    return path
