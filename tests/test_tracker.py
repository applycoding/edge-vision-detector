import numpy as np

from edge_vision.tracker import SortTracker
from edge_vision.types import Detection


def _box(x: float, y: float = 10, w: float = 30, h: float = 40, class_id: int = 0, name: str = "person") -> Detection:
    return Detection((x, y, x + w, y + h), 0.9, class_id, name)


def test_track_is_confirmed_only_after_min_hits() -> None:
    tracker = SortTracker(iou_threshold=0.3, max_age=5, min_hits=3)
    det = _box(10)
    assert tracker.update([det]) == []
    assert tracker.update([det]) == []
    tracks = tracker.update([det])
    assert len(tracks) == 1
    assert tracks[0].track_id == 1
    assert tracks[0].hits == 3
    assert tracks[0].time_since_update == 0
    assert tracks[0].confirmed
    assert tracks[0].class_name == "person"
    np.testing.assert_allclose(tracks[0].bbox, det.bbox, atol=1e-4)


def test_parallel_objects_keep_separate_ids() -> None:
    tracker = SortTracker(iou_threshold=0.3, max_age=5, min_hits=1)
    ids: dict[int, int] = {}
    for step in range(6):
        tracks = tracker.update(
            [
                _box(10 + step * 4, class_id=0, name="person"),
                _box(200 + step * 4, y=180, class_id=2, name="car"),
            ]
        )
        found = {track.class_id: track.track_id for track in tracks}
        if not ids:
            ids = found
        assert found == ids
    assert set(ids) == {0, 2}


def test_track_is_dropped_after_max_age_and_ids_are_not_reused() -> None:
    tracker = SortTracker(iou_threshold=0.3, max_age=2, min_hits=1)
    det = _box(0)
    first = tracker.update([det])
    assert first[0].track_id == 1
    assert len(tracker.update([])) == 1
    coasting = tracker.update([])
    assert len(coasting) == 1
    assert coasting[0].time_since_update == 2
    assert tracker.update([]) == []
    revived = tracker.update([det])
    assert revived[0].track_id == 2


def test_different_class_does_not_inherit_the_track() -> None:
    tracker = SortTracker(iou_threshold=0.3, max_age=3, min_hits=1)
    tracker.update([_box(0, class_id=0, name="person")])
    tracks = tracker.update([_box(0, class_id=2, name="car")])
    by_class = {track.class_id: track for track in tracks}
    assert set(by_class) == {0, 2}
    assert by_class[0].track_id != by_class[2].track_id
    assert by_class[0].time_since_update == 1
    assert by_class[2].time_since_update == 0
