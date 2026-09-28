from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path


APP_NAME = "LAS_CAFIISICA"


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _build_root() -> Path:
    local = os.environ.get("LOCALAPPDATA")
    if local:
        return Path(local) / "LBM" / "LASCAFIISICA"
    return _repo_root() / "builds" / "local_short"


def _run(command: list[str], cwd: Path) -> None:
    print(
        "+",
        " ".join(str(item) for item in command),
        flush=True,
    )
    completed = subprocess.run(command, cwd=cwd)
    if completed.returncode != 0:
        raise SystemExit(completed.returncode)


def _copy_runtime_assets(
    repo: Path,
    app_dist: Path,
) -> None:
    required = [
        repo / "viewer" / "index.html",
        repo / "viewer" / "app.js",
        (
            repo
            / "vendor"
            / "potree"
            / "build"
            / "potree"
            / "potree.js"
        ),
        (
            repo
            / "vendor"
            / "potreeconverter"
            / "PotreeConverter.exe"
        ),
    ]
    missing = [
        str(path)
        for path in required
        if not path.exists()
    ]
    if missing:
        raise RuntimeError(
            "Viewer runtime assets missing:\n"
            + "\n".join(missing)
        )

    shutil.copytree(
        repo / "viewer",
        app_dist / "viewer",
        dirs_exist_ok=True,
    )
    shutil.copytree(
        repo / "vendor" / "potree",
        app_dist / "vendor" / "potree",
        dirs_exist_ok=True,
    )
    shutil.copytree(
        repo / "vendor" / "potreeconverter",
        app_dist / "vendor" / "potreeconverter",
        dirs_exist_ok=True,
    )


def main() -> int:
    if os.name != "nt":
        print(
            "This builder targets Windows only.",
            file=sys.stderr,
        )
        return 2

    repo = _repo_root()
    root = _build_root()
    dist_root = root / "dist"
    work_root = root / "work"
    spec_root = root / "spec"

    if root.exists():
        shutil.rmtree(root)
    dist_root.mkdir(
        parents=True,
        exist_ok=True,
    )
    work_root.mkdir(
        parents=True,
        exist_ok=True,
    )
    spec_root.mkdir(
        parents=True,
        exist_ok=True,
    )

    command = [
        sys.executable,
        "-m",
        "PyInstaller",
        "--noconfirm",
        "--clean",
        "--windowed",
        "--noupx",
        "--name",
        APP_NAME,
        "--distpath",
        str(dist_root),
        "--workpath",
        str(work_root),
        "--specpath",
        str(spec_root),
        "--paths",
        str(repo / "src"),
        "--collect-all",
        "laspy",
        "--collect-all",
        "lazrs",
        "--collect-all",
        "pyproj",
        "--collect-all",
        "scipy",
        "--hidden-import",
        "PySide6.QtWebEngineWidgets",
        "--hidden-import",
        "PySide6.QtWebEngineCore",
        str(repo / "app" / "main.py"),
    ]
    _run(command, cwd=repo)

    app_dist = dist_root / APP_NAME
    exe = app_dist / f"{APP_NAME}.exe"
    if not exe.is_file():
        print(
            f"Executable was not created: {exe}",
            file=sys.stderr,
        )
        return 3

    _copy_runtime_assets(
        repo,
        app_dist,
    )
    print(f"BUILD_OK={exe}")
    print(
        "VIEWER_OK="
        f"{app_dist / 'viewer' / 'index.html'}"
    )
    print(
        "POTREE_CONVERTER_OK="
        f"{app_dist / 'vendor' / 'potreeconverter' / 'PotreeConverter.exe'}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
