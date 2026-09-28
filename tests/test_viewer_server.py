from __future__ import annotations

import urllib.request

from las_classifier.viewer.converter import (
    source_fingerprint,
)
from las_classifier.viewer.server import ViewerServer


def test_viewer_server_serves_app_and_cloud(
    tmp_path,
):
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


def test_source_fingerprint_changes_with_file_content(
    tmp_path,
):
    source = tmp_path / "cloud.las"
    source.write_bytes(b"abc")
    first = source_fingerprint(source)
    source.write_bytes(b"abcd")
    second = source_fingerprint(source)
    assert first != second
