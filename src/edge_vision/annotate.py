from __future__ import annotations

from collections.abc import Sequence

import cv2
import numpy as np

from edge_vision.types import TrackState
from edge_vision.zones import Zone, ZoneAlert


def annotate(
    frame: np.ndarray,
    tracks: Sequence[TrackState],
    zones: Sequence[Zone],
    alerts: Sequence[ZoneAlert],
    *,
    fps: float | None = None,
    latency_ms: float | None = None,
) -> np.ndarray:
    """Draw zones, tracks, and a FPS/latency line. Does not modify `frame`."""
    canvas = frame.copy()
    for zone in zones:
        points = np.asarray(zone.polygon, dtype=np.int32).reshape(-1, 1, 2)
        cv2.polylines(canvas, [points], isClosed=True, color=(0, 165, 255), thickness=2)
        origin = zone.polygon[0]
        _label(canvas, zone.name, int(origin[0]), int(origin[1]), (0, 165, 255))
    entered = {alert.track_id for alert in alerts if alert.kind == "zone_enter"}
    for track in tracks:
        color = (0, 0, 220) if track.track_id in entered else _color(track.track_id)
        x1, y1, x2, y2 = (int(round(value)) for value in track.bbox)
        cv2.rectangle(canvas, (x1, y1), (x2, y2), color, 2)
        name = track.class_name or str(track.class_id)
        _label(canvas, f"{name} #{track.track_id} {track.score:.2f}", x1, y1, color)
    hud = _hud_text(fps, latency_ms)
    if hud:
        _label(canvas, hud, 8, 22, (32, 32, 32))
    return canvas


def _hud_text(fps: float | None, latency_ms: float | None) -> str:
    parts: list[str] = []
    if fps is not None:
        parts.append(f"{fps:.1f} fps")
    if latency_ms is not None:
        parts.append(f"{latency_ms:.0f} ms")
    return "  ".join(parts)


def _label(canvas: np.ndarray, text: str, x: int, y: int, color: tuple[int, int, int]) -> None:
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.5
    thickness = 1
    (width, height), baseline = cv2.getTextSize(text, font, scale, thickness)
    top = max(0, y - height - 6)
    cv2.rectangle(canvas, (x, top), (x + width + 6, top + height + baseline + 4), color, -1)
    cv2.putText(
        canvas,
        text,
        (x + 3, top + height + 1),
        font,
        scale,
        (255, 255, 255),
        thickness,
        cv2.LINE_AA,
    )


def _color(track_id: int) -> tuple[int, int, int]:
    rng = np.random.default_rng(track_id)
    values = rng.integers(50, 230, size=3)
    return int(values[0]), int(values[1]), int(values[2])
