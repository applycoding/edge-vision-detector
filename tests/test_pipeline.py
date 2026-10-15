from pathlib import Path

import cv2
import numpy as np

from edge_vision.annotate import annotate
from edge_vision.capture import ArraySource
from edge_vision.pipeline import Pipeline
from edge_vision.tracker import SortTracker
from edge_vision.types import Detection, StageTiming, TrackState
from edge_vision.zones import Zone, ZoneMonitor


class _Scripted:
    def __init__(self, per_frame: list[list[Detection]]) -> None:
        self._per_frame = per_frame
        self.calls = 0

    def detect(self, frame: np.ndarray) -> tuple[list[Detection], StageTiming]:
        detections = self._per_frame[self.calls] if self.calls < len(self._per_frame) else []
        self.calls += 1
        return detections, StageTiming(0.001, 0.002, 0.003)


class _Recording:
    def __init__(self) -> None:
        self.events: list[dict[str, object]] = []
        self.closed = False

    def publish(self, event: dict[str, object]) -> None:
        self.events.append(event)

    def close(self) -> None:
        self.closed = True


def test_pipeline_publishes_one_enter_and_writes_video(tmp_path: Path) -> None:
    frames = [np.zeros((80, 120, 3), dtype=np.uint8) for _ in range(3)]
    detection = Detection((30, 10, 50, 70), 0.88, 0, "person")
    zone = Zone("dock", ((0, 0), (100, 0), (100, 80), (0, 80)), ("person",))
    publisher = _Recording()
    output = tmp_path / "out.mp4"
    pipeline = Pipeline(
        source=ArraySource(frames, fps=10),
        detector=_Scripted([[detection], [detection], [detection]]),
        tracker=SortTracker(iou_threshold=0.3, max_age=5, min_hits=1),
        zone_monitor=ZoneMonitor([zone]),
        zones=[zone],
        publisher=publisher,
        output_path=str(output),
        show=False,
        metrics_every=0,
    )
    summary = pipeline.run()
    assert summary["frames"] == 3
    assert summary["detections"] == 3
    assert summary["alerts"] == 1
    assert publisher.closed
    assert [event["event"] for event in publisher.events] == ["zone_enter"]
    assert output.stat().st_size > 0
    capture = cv2.VideoCapture(str(output))
    ok, frame = capture.read()
    capture.release()
    assert ok and frame is not None
    assert frame.shape == (80, 120, 3)


def test_warmup_and_max_frames_limit_the_measured_run() -> None:
    frames = [np.zeros((16, 16, 3), dtype=np.uint8) for _ in range(6)]
    detector = _Scripted([[] for _ in frames])
    pipeline = Pipeline(
        source=ArraySource(frames, fps=10),
        detector=detector,
        tracker=SortTracker(min_hits=1),
        zone_monitor=ZoneMonitor([]),
        zones=[],
        publisher=_Recording(),
        output_path="",
    )
    summary = pipeline.run(max_frames=2, warmup=2)
    assert summary["frames"] == 2
    assert detector.calls == 4
    assert "total" in summary["stages_ms"]


def test_annotate_draws_on_a_blank_frame() -> None:
    frame = np.zeros((40, 60, 3), dtype=np.uint8)
    track = TrackState(1, (5, 5, 25, 30), 0.5, 0, "person", 3, 3, 0, True)
    zone = Zone("door", ((0, 0), (30, 0), (30, 30)), ())
    drawn = annotate(frame, [track], [zone], [], fps=12.5, latency_ms=8)
    assert drawn.shape == frame.shape
    assert int(drawn.sum()) > 0
    assert int(frame.sum()) == 0
