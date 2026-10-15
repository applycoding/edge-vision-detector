import pytest

from edge_vision.cli import main


def test_help_and_version() -> None:
    with pytest.raises(SystemExit) as help_exit:
        main(["--help"])
    assert help_exit.value.code == 0
    with pytest.raises(SystemExit) as version_exit:
        main(["--version"])
    assert version_exit.value.code == 0


def test_missing_model_returns_1(tmp_path) -> None:
    config = tmp_path / "demo.yaml"
    config.write_text(
        "source:\n  kind: file\n  path: /tmp/unused.mp4\nmodel:\n  path: /tmp/missing-yolo.onnx\n",
        encoding="utf-8",
    )
    assert main(["run", "--config", str(config), "--max-frames", "1"]) == 1
