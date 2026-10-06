import json
from pathlib import Path


EXPECTED = "LAS_CAFIISICA_GROUND_V2_2026_10_R20_6_2"


def _load(path: str) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def test_localbuild_source_guard_matches_source_revision():
    actual = Path("localbuild/SOURCE_REVISION.txt").read_text(encoding="utf-8").strip()
    assert actual == EXPECTED

    root = _load("localbuild/las_cafiisica.json")
    bundled = _load(
        "tools/LocalBuildManager_CLASSIFY_LAS/projects/"
        "las_cafiisica_ground_v2.json"
    )

    for cfg in (root, bundled):
        assert cfg["branch"] == "r20-6-2-ground-complete-veto-mdt"
        assert int(cfg["config_revision"]) >= 54
        guard = cfg["required_source_revision"]
        assert guard["path"] == "localbuild/SOURCE_REVISION.txt"
        assert guard["value"] == EXPECTED

        verify = next(
            step for step in cfg["pipelines"]["test"]
            if step["id"] == "verify-ground-v2-source"
        )
        assert f"$expected='{EXPECTED}';" in verify["command"]
        assert EXPECTED + "_2" not in verify["command"]


def test_root_and_bundled_profiles_require_same_revision():
    root = _load("localbuild/las_cafiisica.json")
    bundled = _load(
        "tools/LocalBuildManager_CLASSIFY_LAS/projects/"
        "las_cafiisica_ground_v2.json"
    )
    assert root["required_source_revision"] == bundled["required_source_revision"]
    assert root["config_revision"] == bundled["config_revision"]
