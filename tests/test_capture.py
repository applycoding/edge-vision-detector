from pathlib import Path

import cv2
import numpy as np
import pytest

from edge_vision.capture import open_source
from edge_vision.config import SourceConfig
from tests.conftest import ROOT


def test_file_source_reads_written_frames(tmp_path: Path) -> None:
    path = tmp_path / "clip.mp4"
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 10, (32, 24))
    assert writer.isOpened()
    for index in range(3):
        writer.write(np.full((24, 32, 3), index * 20, dtype=np.uint8))
    writer.release()
    source = open_source(SourceConfig(kind="file", path=str(path)))
    frames = list(source.frames())
    source.close()
    source.close()
    assert len(frames) == 3
    assert frames[0].shape == (24, 32, 3)


def test_missing_file_raises() -> None:
    with pytest.raises(RuntimeError):
        open_source(SourceConfig(kind="file", path="/tmp/edge-vision-missing.mp4"))


def test_checked_in_sample_clip_opens() -> None:
    path = ROOT / "samples" / "synthetic.mp4"
    assert path.is_file()
    assert (ROOT / "samples" / "frames" / "frame_000.png").is_file()
    source = open_source(SourceConfig(kind="file", path=str(path)))
    frame = next(source.frames())
    source.close()
    assert frame.shape[0] == 480
    assert frame.shape[1] == 640
