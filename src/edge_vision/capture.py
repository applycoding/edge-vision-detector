from __future__ import annotations

from collections.abc import Iterator, Sequence
from typing import Protocol

import cv2
import numpy as np

from edge_vision.config import SourceConfig


class FrameSource(Protocol):
    fps: float

    def frames(self) -> Iterator[np.ndarray]: ...

    def close(self) -> None: ...


class VideoSource:
    def __init__(self, capture: cv2.VideoCapture, fps: float) -> None:
        self._capture = capture
        self.fps = fps if fps and fps > 1.0 else 30.0
        self._closed = False

    def frames(self) -> Iterator[np.ndarray]:
        while True:
            ok, frame = self._capture.read()
            if not ok:
                break
            yield frame

    def close(self) -> None:
        if self._closed:
            return
        self._capture.release()
        self._closed = True


class ArraySource:
    """In-memory frames for tests and the synthetic demo path."""

    def __init__(self, frames: Sequence[np.ndarray], fps: float = 30.0) -> None:
        self._frames = list(frames)
        self.fps = fps if fps > 1.0 else 30.0

    def frames(self) -> Iterator[np.ndarray]:
        yield from self._frames

    def close(self) -> None:
        return None


def open_source(cfg: SourceConfig) -> VideoSource:
    kind = cfg.kind
    if kind == "file":
        capture = cv2.VideoCapture(cfg.path)
        label = cfg.path
    elif kind == "camera":
        capture = cv2.VideoCapture(cfg.device)
        label = f"camera:{cfg.device}"
    elif kind == "rtsp":
        capture = cv2.VideoCapture(cfg.url, cv2.CAP_FFMPEG)
        capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        label = cfg.url
    else:
        _never: never = kind
        raise ValueError(f"unsupported source kind: {_never}")
    if not capture.isOpened():
        capture.release()
        raise RuntimeError(f"could not open {kind} source ({label})")
    return VideoSource(capture, fps=float(capture.get(cv2.CAP_PROP_FPS) or 0.0))
