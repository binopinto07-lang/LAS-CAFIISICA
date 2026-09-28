from __future__ import annotations

from pathlib import Path


def _app_js() -> str:
    root = Path(__file__).resolve().parents[1]
    return (root / "viewer" / "app.js").read_text(encoding="utf-8")


def test_view_mode_switch_does_not_reframe_camera():
    source = _app_js()
    start = source.index("function setViewMode(mode)")
    stop = source.index("function fit()", start)
    block = source[start:stop]

    assert "fitRepeatedly();" not in block


def test_only_first_loaded_cloud_is_auto_fitted():
    source = _app_js()

    assert "autoFitted: false" in source
    assert "if (!state.autoFitted)" in source
    assert "state.autoFitted = true;" in source
    assert "state.autoFitted = false;" in source
