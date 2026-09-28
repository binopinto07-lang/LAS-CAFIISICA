from __future__ import annotations

import logging
import mimetypes
import re
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import unquote, urlsplit

from .paths import potree_root, viewer_root


LOGGER = logging.getLogger("las_cafiisica.viewer.server")
_RANGE_RE = re.compile(r"^bytes=(\d*)-(\d*)$")


class ViewerServer:
    def __init__(
        self,
        viewer_dir: Path | None = None,
        potree_dir: Path | None = None,
    ) -> None:
        self.viewer_dir = (viewer_dir or viewer_root()).resolve()
        self.potree_dir = (potree_dir or potree_root()).resolve()
        self._clouds: dict[str, Path] = {}
        self._lock = threading.RLock()
        self._httpd: ThreadingHTTPServer | None = None
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        if self._httpd is not None:
            return

        owner = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, fmt: str, *args) -> None:
                LOGGER.debug("HTTP " + fmt, *args)

            def do_HEAD(self) -> None:
                self._serve(send_body=False)

            def do_GET(self) -> None:
                self._serve(send_body=True)

            def _serve(self, send_body: bool) -> None:
                try:
                    target = owner.resolve_url_path(self.path)
                except (FileNotFoundError, PermissionError, ValueError):
                    self.send_error(404)
                    return

                content_type = (
                    mimetypes.guess_type(target.name)[0]
                    or "application/octet-stream"
                )
                size = target.stat().st_size

                try:
                    byte_range = self._requested_range(size)
                except ValueError:
                    self.send_response(416)
                    self.send_header("Content-Range", f"bytes */{size}")
                    self.end_headers()
                    return

                if byte_range is None:
                    start = 0
                    end = max(0, size - 1)
                    status = 200
                else:
                    start, end = byte_range
                    status = 206

                length = 0 if size == 0 else end - start + 1
                self.send_response(status)
                self.send_header("Content-Type", content_type)
                self.send_header("Accept-Ranges", "bytes")
                self.send_header("Content-Length", str(length))
                self.send_header("Cache-Control", "no-cache")
                if status == 206:
                    self.send_header(
                        "Content-Range",
                        f"bytes {start}-{end}/{size}",
                    )
                self.end_headers()

                if not send_body or length == 0:
                    return

                try:
                    with target.open("rb") as stream:
                        stream.seek(start)
                        remaining = length
                        while remaining > 0:
                            chunk = stream.read(min(1024 * 1024, remaining))
                            if not chunk:
                                break
                            self.wfile.write(chunk)
                            remaining -= len(chunk)
                except (BrokenPipeError, ConnectionResetError):
                    return

            def _requested_range(
                self,
                size: int,
            ) -> tuple[int, int] | None:
                header = self.headers.get("Range")
                if not header:
                    return None

                match = _RANGE_RE.match(header.strip())
                if match is None or size <= 0:
                    raise ValueError("Invalid Range")

                first, last = match.groups()
                if first == "" and last == "":
                    raise ValueError("Invalid Range")

                if first == "":
                    suffix = int(last)
                    if suffix <= 0:
                        raise ValueError("Invalid suffix Range")
                    suffix = min(suffix, size)
                    return size - suffix, size - 1

                start = int(first)
                if start >= size:
                    raise ValueError("Range starts beyond EOF")

                if last == "":
                    end = size - 1
                else:
                    end = min(int(last), size - 1)

                if end < start:
                    raise ValueError("Invalid Range")
                return start, end

        self._httpd = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self._thread = threading.Thread(
            target=self._httpd.serve_forever,
            daemon=True,
            name="las-cafiisica-viewer-http",
        )
        self._thread.start()
        LOGGER.info("VIEWER_SERVER=%s", self.base_url)

    @property
    def base_url(self) -> str:
        if self._httpd is None:
            raise RuntimeError("Viewer server is not running")
        host, port = self._httpd.server_address[:2]
        return f"http://{host}:{port}"

    @property
    def app_url(self) -> str:
        return self.base_url + "/app/"

    def register_cloud(self, key: str, directory: str | Path) -> str:
        clean_key = self._clean_key(key)
        root = Path(directory).resolve()
        metadata = root / "metadata.json"
        if not metadata.is_file():
            raise FileNotFoundError(metadata)
        with self._lock:
            self._clouds[clean_key] = root
        return self.base_url + f"/cloud/{clean_key}/metadata.json"

    def resolve_url_path(self, raw_url: str) -> Path:
        path = unquote(urlsplit(raw_url).path)
        if path in {"/", "/app", "/app/"}:
            return self._safe_file(self.viewer_dir, "index.html")
        if path.startswith("/app/"):
            return self._safe_file(
                self.viewer_dir,
                path[len("/app/"):],
            )
        if path.startswith("/potree/"):
            return self._safe_file(
                self.potree_dir,
                path[len("/potree/"):],
            )
        if path.startswith("/cloud/"):
            rest = path[len("/cloud/"):]
            key, separator, rel = rest.partition("/")
            if not separator:
                raise FileNotFoundError(path)
            clean_key = self._clean_key(key)
            with self._lock:
                root = self._clouds.get(clean_key)
            if root is None:
                raise FileNotFoundError(path)
            return self._safe_file(root, rel)
        raise FileNotFoundError(path)

    @staticmethod
    def _safe_file(root: Path, relative: str) -> Path:
        root = root.resolve()
        candidate = (root / relative).resolve()
        if root != candidate and root not in candidate.parents:
            raise PermissionError(relative)
        if not candidate.is_file():
            raise FileNotFoundError(candidate)
        return candidate

    @staticmethod
    def _clean_key(key: str) -> str:
        clean = "".join(
            char
            for char in str(key).lower()
            if char.isalnum() or char in {"-", "_"}
        )
        if not clean:
            raise ValueError("Invalid cloud key")
        return clean

    def stop(self) -> None:
        httpd = self._httpd
        self._httpd = None
        if httpd is not None:
            httpd.shutdown()
            httpd.server_close()
        thread = self._thread
        self._thread = None
        if thread is not None and thread.is_alive():
            thread.join(timeout=2.0)
