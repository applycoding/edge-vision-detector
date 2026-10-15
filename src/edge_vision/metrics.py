from __future__ import annotations

import time
from typing import Any

import numpy as np


class RunningMetrics:
    """Wall-clock FPS plus per-stage latency samples, in seconds internally."""

    def __init__(self) -> None:
        self.frame_count = 0
        self.detections = 0
        self.alerts = 0
        self.frame_size: tuple[int, int] | None = None
        self._started = time.perf_counter()
        self._stages: dict[str, list[float]] = {
            "preprocess": [],
            "inference": [],
            "postprocess": [],
            "track": [],
            "total": [],
        }

    def add(self, stage: str, seconds: float) -> None:
        self._stages[stage].append(float(seconds))

    def reset(self) -> None:
        self.frame_count = 0
        self.detections = 0
        self.alerts = 0
        self._started = time.perf_counter()
        for samples in self._stages.values():
            samples.clear()

    @property
    def fps(self) -> float:
        elapsed = time.perf_counter() - self._started
        if elapsed <= 0 or self.frame_count == 0:
            return 0.0
        return self.frame_count / elapsed

    def summary(self) -> dict[str, Any]:
        elapsed = time.perf_counter() - self._started
        fps = self.frame_count / elapsed if elapsed > 0 and self.frame_count else 0.0
        stages: dict[str, dict[str, float]] = {}
        for name, samples in self._stages.items():
            if not samples:
                continue
            values = np.asarray(samples, dtype=np.float64) * 1000.0
            stages[name] = {
                "mean": float(values.mean()),
                "p50": float(np.percentile(values, 50)),
                "p95": float(np.percentile(values, 95)),
            }
        report: dict[str, Any] = {
            "frames": self.frame_count,
            "elapsed_s": float(elapsed),
            "fps": float(fps),
            "detections": self.detections,
            "alerts": self.alerts,
            "stages_ms": stages,
        }
        if self.frame_size is not None:
            report["frame_size"] = [self.frame_size[0], self.frame_size[1]]
        return report


def format_report(summary: dict[str, Any]) -> str:
    lines = [
        f"frames: {summary['frames']}",
        f"fps: {summary['fps']:.2f}",
        f"detections: {summary['detections']}",
        f"alerts: {summary['alerts']}",
    ]
    size = summary.get("frame_size")
    if size:
        lines.append(f"frame_size: {size[0]}x{size[1]}")
    stages = summary.get("stages_ms", {})
    for name in ("preprocess", "inference", "postprocess", "track", "total"):
        stats = stages.get(name)
        if not stats:
            continue
        lines.append(
            f"{name}: mean {stats['mean']:.2f} ms, p50 {stats['p50']:.2f} ms, p95 {stats['p95']:.2f} ms"
        )
    return "\n".join(lines)
