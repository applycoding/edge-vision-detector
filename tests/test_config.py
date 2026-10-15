import pytest

from edge_vision.config import load_config
from tests.conftest import ROOT


def test_default_config() -> None:
    cfg = load_config(ROOT / "configs" / "default.yaml")
    assert cfg.source.kind == "file"
    assert cfg.source.path == "samples/synthetic.mp4"
    assert cfg.model.input_size == (640, 640)
    assert cfg.model.providers == ("CPUExecutionProvider",)
    assert cfg.model.layout == "auto"
    assert cfg.tracker.min_hits == 3
    assert [zone.name for zone in cfg.zones] == ["doorway"]
    assert cfg.zones[0].classes == ("person",)
    assert cfg.publish.mode == "none"
    assert cfg.max_frames is None


def test_camera_and_rtsp_configs() -> None:
    camera = load_config(ROOT / "configs" / "camera.yaml")
    rtsp = load_config(ROOT / "configs" / "rtsp.yaml")
    assert camera.source.kind == "camera"
    assert camera.source.device == 0
    assert rtsp.source.kind == "rtsp"
    assert rtsp.source.url.startswith("rtsp://")
    assert rtsp.publish.mode == "mqtt"
    assert rtsp.publish.mqtt_topic == "edge/alerts"


def test_unknown_field_and_missing_source(tmp_path) -> None:
    broken = tmp_path / "broken.yaml"
    broken.write_text("sorce:\n  kind: file\n", encoding="utf-8")
    with pytest.raises(ValueError, match="unknown"):
        load_config(broken)
    empty = tmp_path / "empty.yaml"
    empty.write_text("{}\n", encoding="utf-8")
    with pytest.raises(ValueError, match="source"):
        load_config(empty)
    missing = tmp_path / "nope.yaml"
    with pytest.raises(FileNotFoundError):
        load_config(missing)
