from edge_vision.types import TrackState
from edge_vision.zones import Zone, ZoneMonitor, foot_point, point_in_polygon


def _track(bbox: tuple[float, float, float, float], *, confirmed: bool = True, class_name: str = "person", class_id: int = 0, track_id: int = 1) -> TrackState:
    return TrackState(
        track_id=track_id,
        bbox=bbox,
        score=0.9,
        class_id=class_id,
        class_name=class_name,
        hits=3,
        age=3,
        time_since_update=0,
        confirmed=confirmed,
    )


def test_point_in_polygon_and_foot_point() -> None:
    square = ((0.0, 0.0), (10.0, 0.0), (10.0, 10.0), (0.0, 10.0))
    assert point_in_polygon(5, 5, square)
    assert not point_in_polygon(15, 5, square)
    assert not point_in_polygon(-1, 5, square)
    assert foot_point((0, 0, 10, 20)) == (5, 20)


def test_enter_is_edge_triggered_on_the_foot_point() -> None:
    # The centroid of (0, 0, 10, 20) is outside this polygon; the foot is inside.
    zone = Zone("floor", ((0, 15), (20, 15), (20, 30), (0, 30)), ("person",))
    monitor = ZoneMonitor([zone], clock=lambda: "2026-10-15T00:00:00.000Z")
    first = monitor.update([_track((0, 0, 10, 20))])
    assert [alert.kind for alert in first] == ["zone_enter"]
    payload = first[0].to_dict()
    assert payload["zone"] == "floor"
    assert payload["timestamp"] == "2026-10-15T00:00:00.000Z"
    assert payload["class_name"] == "person"
    assert monitor.update([_track((0, 0, 10, 20))]) == []
    left = monitor.update([_track((0, 0, 10, 5))])
    assert [alert.kind for alert in left] == ["zone_exit"]


def test_class_filter_unconfirmed_tracks_and_disappearance() -> None:
    zone = Zone("door", ((0, 0), (50, 0), (50, 50), (0, 50)), ("car", 2))
    monitor = ZoneMonitor([zone], clock=lambda: "T")
    assert monitor.update([_track((10, 10, 20, 40), class_name="person", class_id=0)]) == []
    assert monitor.update([_track((10, 10, 20, 40), class_name="car", class_id=2, confirmed=False)]) == []
    entered = monitor.update([_track((10, 10, 20, 40), class_name="car", class_id=2)])
    assert [alert.kind for alert in entered] == ["zone_enter"]
    assert monitor.update([]) == []
    again = monitor.update([_track((10, 10, 20, 40), class_name="car", class_id=2)])
    assert [alert.kind for alert in again] == ["zone_enter"]
