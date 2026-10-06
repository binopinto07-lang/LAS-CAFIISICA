from pathlib import Path


def test_windows_build_collects_rasterio_runtime():
    source = Path("scripts/build_windows.py").read_text(encoding="utf-8")
    assert '"--collect-all",\n        "rasterio"' in source


def test_r20_6_2_is_default_engine_and_mdt_requires_preview_before_export():
    ui = Path("src/las_classifier/main_window.py").read_text(encoding="utf-8")
    worker = Path("src/las_classifier/cloud/classification_worker.py").read_text(encoding="utf-8")
    assert 'self.engine_combo.setCurrentText("Universal Ground R20.6.2")' in ui
    assert 'QPushButton("CRIAR MDT (PREVIEW)")' in ui
    assert 'QPushButton("EXPORTAR MDT VALIDADO")' in ui
    assert "self._mdt_preview is not None" in ui
    assert 'self.engine_name == "Universal Ground R20.6.2"' in worker


def test_self_test_exercises_mdt_and_rasterio():
    source = Path("src/las_classifier/self_test.py").read_text(encoding="utf-8")
    assert "import rasterio" in source
    assert "MDT_PREVIEW_IN_MEMORY" in source
    assert "MDT_GEOTIFF" in source
    assert "MDT_OBSERVATION_STATE" in source
