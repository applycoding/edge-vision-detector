from __future__ import annotations

import logging
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Protocol

import cv2
import numpy as np

from edge_vision.annotate import annotate
from edge_vision.capture import FrameSource
from edge_vision.metrics import RunningMetrics
from edge_vision.tracker import SortTracker
from edge_vision.types import Detection, StageTiming
from edge_vision.zones import Zone, ZoneMonitor

logger = logging.getLogger(__name__)


class Detector(Protocol):
    def detect(self, frame: np.ndarray) -> tuple[list[Detection], StageTiming]: ...


class EventPublisher(Protocol):
    def publish(self, event: dict[str, object]) -> None: ...

    def close(self) -> None: ...


class Pipeline:
    def __init__(
        self,
        source: FrameSource,
        detector: Detector,
        tracker: SortTracker,
        zone_monitor: ZoneMonitor,
        zones: Sequence[Zone],
        publisher: EventPublisher,
        output_path: str = "",
        show: bool = False,
        metrics_every: int = 30,
    ) -> None:
        self.source = source
        self.detector = detector
        self.tracker = tracker
        self.zone_monitor = zone_monitor
        self.zones = tuple(zones)
        self.publisher = publisher
        self.output_path = output_path
        self.show = show
        self.metrics_every = metrics_every

    def run(self, max_frames: int | None = None, warmup: int = 0) -> dict[str, object]:
        """Process frames. `warmup` frames update tracker state but are left out of the metrics."""
        if warmup < 0:
            raise ValueError("warmup must be >= 0")
        metrics = RunningMetrics()
        sink = _VideoSink()
        seen = 0
        measured = 0
        try:
            for frame in self.source.frames():
                warming = seen < warmup
                if not warming and max_frames is not None and measured >= max_frames:
                    break
                seen += 1
                latency_s = self._process_frame(frame, metrics, sink)
                if warming:
                    if seen == warmup:
                        metrics.reset()
                    continue
                measured += 1
                if self.metrics_every and measured % self.metrics_every == 0:
                    logger.info(
                        "frames=%d fps=%.2f latency=%.1fms",
                        metrics.frame_count,
                        metrics.fps,
                        latency_s * 1000.0,
                    )
        finally:
            self.source.close()
            if sink.writer is not None:
                sink.writer.release()
            self.publisher.close()
            if self.show:
                cv2.destroyAllWindows()
        if max_frames and measured < max_frames:
            logger.warning("source ended after %d measured frames (requested %d)", measured, max_frames)
        return metrics.summary()

    def _process_frame(self, frame: np.ndarray, metrics: RunningMetrics, sink: _VideoSink) -> float:
        started = time.perf_counter()
        detections, stage = self.detector.detect(frame)
        tracked_at = time.perf_counter()
        tracks = self.tracker.update(detections)
        alerts = self.zone_monitor.update(tracks)
        for alert in alerts:
            self.publisher.publish(alert.to_dict())
        tracked = time.perf_counter()
        latency_s = tracked - started
        annotated = annotate(
            frame,
            tracks,
            self.zones,
            alerts,
            fps=metrics.fps,
            latency_ms=latency_s * 1000.0,
        )
        if metrics.frame_size is None:
            metrics.frame_size = (frame.shape[1], frame.shape[0])
        _write_frame(sink, annotated, self.output_path, self.source.fps)
        if self.show:
            _show(annotated)
        finished = time.perf_counter()
        metrics.frame_count += 1
        metrics.detections += len(detections)
        metrics.alerts += len(alerts)
        metrics.add("preprocess", stage.preprocess_s)
        metrics.add("inference", stage.inference_s)
        metrics.add("postprocess", stage.postprocess_s)
        metrics.add("track", tracked - tracked_at)
        metrics.add("total", finished - started)
        return latency_s


class _VideoSink:
    def __init__(self) -> None:
        self.writer: cv2.VideoWriter | None = None
        self.size: tuple[int, int] | None = None


def _write_frame(sink: _VideoSink, frame: np.ndarray, output_path: str, fps: float) -> None:
    if not output_path:
        return
    height, width = frame.shape[:2]
    if sink.writer is None:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        sink.writer = _open_writer(output_path, fps, (width, height))
        sink.size = (width, height)
    elif sink.size != (width, height):
        raise RuntimeError(f"frame size changed from {sink.size} to {width}x{height}")
    sink.writer.write(frame)


def _open_writer(path: str, fps: float, size: tuple[int, int]) -> cv2.VideoWriter:
    width, height = size
    suffix = Path(path).suffix.lower()
    codecs = ("mp4v", "avc1") if suffix == ".mp4" else ("MJPG", "XVID")
    rate = fps if fps > 1.0 else 30.0
    for codec in codecs:
        writer = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*codec), rate, (width, height))
        if writer.isOpened():
            return writer
        writer.release()
    raise RuntimeError(f"could not open a video writer for {path}")


def _show(frame: np.ndarray) -> None:
    try:
        cv2.imshow("edge-vision", frame)
        cv2.waitKey(1)
    except cv2.error as exc:
        raise RuntimeError("display output needs a GUI; set output.show to false") from exc

