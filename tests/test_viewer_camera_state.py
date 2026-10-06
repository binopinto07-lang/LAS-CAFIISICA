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


def test_r20_mantle_view_preserves_camera_and_is_not_in_compare_overlay():
    source = _app_js()
    start = source.index("function setViewMode(mode)")
    stop = source.index("function fit()", start)
    block = source[start:stop]

    assert '"mantle"' in block
    assert "fitRepeatedly();" not in block
    assert 'key === "original" || key === "classified"' in source
    assert 'mode === "mantle"' in block


def test_r20_mantle_view_is_lazily_prepared():
    root = Path(__file__).resolve().parents[1]
    widget = (root / "src/las_classifier/viewer/widget.py").read_text(encoding="utf-8")
    worker = (root / "src/las_classifier/viewer/workers.py").read_text(encoding="utf-8")
    assert "mantle_requested = Signal()" in widget
    assert 'self._loaded["mantle"]' in widget
    assert 'self.mantle_requested.emit()' in widget
    assert 'prepare_mantle(' in worker


def test_r20_5_mdt_preview_preserves_camera_and_requires_no_export():
    source = _app_js()
    start = source.index("function showMDTPreview(data)")
    stop = source.index("function setMDTColorMode", start)
    block = source[start:stop]

    assert "fitRepeatedly();" not in block
    assert 'setViewMode("mdt")' in block
    assert "showMDTPreview" in source
    assert "clearMDTPreview: removeMDT" in source


def test_r20_5_mdt_view_has_three_display_modes():
    source = _app_js()
    assert '"elevation", "hillshade", "observation"' in source
    assert "INTERPOLATED" not in source or "observation" in source
