from __future__ import annotations

import urllib.error
import urllib.request

from las_classifier.viewer.converter import source_fingerprint
from las_classifier.viewer.server import ViewerServer


def test_viewer_server_serves_app_and_cloud(tmp_path):
    viewer = tmp_path / "viewer"
    potree = tmp_path / "potree"
    cloud = tmp_path / "cloud"
    viewer.mkdir()
    potree.mkdir()
    cloud.mkdir()
    (viewer / "index.html").write_text(
        "viewer-ok",
        encoding="utf-8",
    )
    (potree / "asset.js").write_text(
        "potree-ok",
        encoding="utf-8",
    )
    (cloud / "metadata.json").write_text(
        "{}",
        encoding="utf-8",
    )

    server = ViewerServer(viewer, potree)
    server.start()
    try:
        cloud_url = server.register_cloud(
            "original",
            cloud,
        )
        assert urllib.request.urlopen(
            server.app_url
        ).read() == b"viewer-ok"
        assert urllib.request.urlopen(
            server.base_url + "/potree/asset.js"
        ).read() == b"potree-ok"
        assert urllib.request.urlopen(
            cloud_url
        ).read() == b"{}"
    finally:
        server.stop()


def test_viewer_server_supports_http_byte_ranges(tmp_path):
    viewer = tmp_path / "viewer"
    potree = tmp_path / "potree"
    cloud = tmp_path / "cloud"
    viewer.mkdir()
    potree.mkdir()
    cloud.mkdir()
    (viewer / "index.html").write_text("ok", encoding="utf-8")
    (cloud / "metadata.json").write_text("{}", encoding="utf-8")
    (cloud / "octree.bin").write_bytes(b"0123456789")

    server = ViewerServer(viewer, potree)
    server.start()
    try:
        server.register_cloud("original", cloud)
        request = urllib.request.Request(
            server.base_url + "/cloud/original/octree.bin",
            headers={"Range": "bytes=2-5"},
        )
        with urllib.request.urlopen(request) as response:
            assert response.status == 206
            assert response.headers["Accept-Ranges"] == "bytes"
            assert response.headers["Content-Range"] == "bytes 2-5/10"
            assert response.read() == b"2345"

        invalid = urllib.request.Request(
            server.base_url + "/cloud/original/octree.bin",
            headers={"Range": "bytes=99-"},
        )
        try:
            urllib.request.urlopen(invalid)
        except urllib.error.HTTPError as exc:
            assert exc.code == 416
        else:
            raise AssertionError("Expected HTTP 416")
    finally:
        server.stop()


def test_source_fingerprint_changes_with_file_content(tmp_path):
    source = tmp_path / "cloud.las"
    source.write_bytes(b"abc")
    first = source_fingerprint(source)
    source.write_bytes(b"abcd")
    second = source_fingerprint(source)
    assert first != second
