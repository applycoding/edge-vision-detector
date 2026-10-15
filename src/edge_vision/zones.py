from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Literal

from edge_vision.types import TrackState

AlertKind = Literal["zone_enter", "zone_exit"]


@dataclass(frozen=True)
class Zone:
    name: str
    polygon: tuple[tuple[float, float], ...]
    classes: tuple[int | str, ...] = ()


@dataclass(frozen=True)
class ZoneAlert:
    kind: AlertKind
    zone: str
    track_id: int
    class_id: int
    class_name: str
    score: float
    bbox: tuple[float, float, float, float]
    timestamp: str

    def to_dict(self) -> dict[str, object]:
        return {
            "event": self.kind,
            "zone": self.zone,
            "track_id": self.track_id,
            "class_id": self.class_id,
            "class_name": self.class_name,
            "score": self.score,
            "bbox": [float(value) for value in self.bbox],
            "timestamp": self.timestamp,
        }


class ZoneMonitor:
    """Emit one enter and one exit when a confirmed track's foot point crosses a polygon.

    The foot point is the bottom-center of the box. A track that disappears
    while inside a zone does not emit exit; the next time that id is seen
    inside, a new enter is emitted. Class filters accept COCO names or class ids.
    An empty filter matches every class.
    """

    def __init__(self, zones: Sequence[Zone], clock: Callable[[], str] | None = None) -> None:
        self.zones = tuple(zones)
        self._clock = clock or utc_timestamp
        self._inside: dict[tuple[int, str], bool] = {}

    def update(self, tracks: Sequence[TrackState]) -> list[ZoneAlert]:
        alerts: list[ZoneAlert] = []
        seen: set[tuple[int, str]] = set()
        timestamp = self._clock()
        for track in tracks:
            if not track.confirmed:
                continue
            foot_x, foot_y = foot_point(track.bbox)
            for zone in self.zones:
                if not _class_selected(track, zone):
                    continue
                key = (track.track_id, zone.name)
                seen.add(key)
                inside = point_in_polygon(foot_x, foot_y, zone.polygon)
                was_inside = self._inside.get(key, False)
                if inside and not was_inside:
                    alerts.append(_alert("zone_enter", zone.name, track, timestamp))
                elif was_inside and not inside:
                    alerts.append(_alert("zone_exit", zone.name, track, timestamp))
                self._inside[key] = inside
        for key in list(self._inside):
            if key not in seen:
                del self._inside[key]
        return alerts


def foot_point(bbox: tuple[float, float, float, float]) -> tuple[float, float]:
    x1, _y1, x2, y2 = bbox
    return ((x1 + x2) / 2.0, y2)


def point_in_polygon(x: float, y: float, polygon: Sequence[tuple[float, float]]) -> bool:
    """Even-odd ray cast. Boundary points are not treated as a special case."""
    inside = False
    previous = len(polygon) - 1
    for index, (xi, yi) in enumerate(polygon):
        xj, yj = polygon[previous]
        crosses = (yi > y) != (yj > y)
        if crosses:
            x_cross = (xj - xi) * (y - yi) / (yj - yi) + xi
            if x < x_cross:
                inside = not inside
        previous = index
    return inside


def utc_timestamp() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")


def _class_selected(track: TrackState, zone: Zone) -> bool:
    if not zone.classes:
        return True
    for item in zone.classes:
        if isinstance(item, str):
            if item.casefold() == track.class_name.casefold():
                return True
            continue
        if item == track.class_id:
            return True
    return False


def _alert(kind: AlertKind, zone: str, track: TrackState, timestamp: str) -> ZoneAlert:
    return ZoneAlert(
        kind=kind,
        zone=zone,
        track_id=track.track_id,
        class_id=track.class_id,
        class_name=track.class_name,
        score=track.score,
        bbox=track.bbox,
        timestamp=timestamp,
    )
