from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from edge_vision.geometry import box_iou_matrix, cxcywh_to_xyxy, xyxy_to_cxcywh
from edge_vision.types import Detection, TrackState


class SortTracker:
    """SORT-style tracker: constant-velocity Kalman filter and greedy IoU association.

    Association is class-aware and greedy in IoU. It is not the Hungarian
    assignment used by the original SORT paper, so two heavily overlapping
    tracks can swap. `update` returns confirmed tracks only. A confirmed track
    is kept, using its predicted box, for `max_age` missed frames.
    """

    def __init__(self, iou_threshold: float = 0.3, max_age: int = 30, min_hits: int = 3) -> None:
        if not 0.0 <= iou_threshold <= 1.0:
            raise ValueError("iou_threshold must be between 0 and 1")
        if max_age < 0:
            raise ValueError("max_age must be >= 0")
        if min_hits < 1:
            raise ValueError("min_hits must be >= 1")
        self.iou_threshold = float(iou_threshold)
        self.max_age = int(max_age)
        self.min_hits = int(min_hits)
        self._tracks: list[_Track] = []
        self._next_id = 1

    def update(self, detections: Sequence[Detection]) -> list[TrackState]:
        for track in self._tracks:
            track.predict()
        matches, _unmatched_tracks, unmatched_dets = greedy_match(
            self._association(detections), self.iou_threshold
        )
        for track_index, det_index in matches:
            self._tracks[track_index].update(detections[det_index])
        for det_index in unmatched_dets:
            self._tracks.append(_Track(self._next_id, detections[det_index]))
            self._next_id += 1
        self._tracks = [track for track in self._tracks if track.time_since_update <= self.max_age]
        return [track.snapshot(self.min_hits) for track in self._tracks if track.hits >= self.min_hits]

    def _association(self, detections: Sequence[Detection]) -> np.ndarray:
        if not self._tracks or not detections:
            return np.zeros((len(self._tracks), len(detections)), dtype=np.float32)
        track_boxes = np.asarray([track.bbox for track in self._tracks], dtype=np.float32)
        det_boxes = np.asarray([detection.bbox for detection in detections], dtype=np.float32)
        iou = box_iou_matrix(track_boxes, det_boxes)
        for track_index, track in enumerate(self._tracks):
            for det_index, detection in enumerate(detections):
                if detection.class_id != track.class_id:
                    iou[track_index, det_index] = 0.0
        return iou


def greedy_match(
    iou: np.ndarray, iou_threshold: float
) -> tuple[list[tuple[int, int]], list[int], list[int]]:
    """One-to-one match, highest IoU first. Pairs below the threshold are ignored."""
    rows, cols = iou.shape
    pairs = [
        (float(iou[row, col]), row, col)
        for row in range(rows)
        for col in range(cols)
        if iou[row, col] >= iou_threshold
    ]
    pairs.sort(key=lambda item: item[0], reverse=True)
    matched_rows: set[int] = set()
    matched_cols: set[int] = set()
    matches: list[tuple[int, int]] = []
    for _score, row, col in pairs:
        if row in matched_rows or col in matched_cols:
            continue
        matched_rows.add(row)
        matched_cols.add(col)
        matches.append((row, col))
    unmatched_rows = [row for row in range(rows) if row not in matched_rows]
    unmatched_cols = [col for col in range(cols) if col not in matched_cols]
    return matches, unmatched_rows, unmatched_cols


class _KalmanBox:
    """8-state constant-velocity filter: cx, cy, w, h and their velocities."""

    def __init__(self, bbox: tuple[float, float, float, float]) -> None:
        self.x = np.zeros((8, 1), dtype=np.float64)
        self.x[:4, 0] = xyxy_to_cxcywh(np.asarray(bbox, dtype=np.float64))
        self.f = np.eye(8, dtype=np.float64)
        for index in range(4):
            self.f[index, index + 4] = 1.0
        self.h = np.eye(4, 8, dtype=np.float64)
        self.p = np.eye(8, dtype=np.float64) * 10.0
        self.p[4:, 4:] *= 100.0
        self.q = np.eye(8, dtype=np.float64)
        self.q[4:, 4:] *= 0.01
        self.r = np.eye(4, dtype=np.float64)
        self.r[2:, 2:] *= 10.0

    def predict(self) -> tuple[float, float, float, float]:
        self.x = self.f @ self.x
        self.p = self.f @ self.p @ self.f.T + self.q
        self._clamp_size()
        return self.bbox()

    def update(self, bbox: tuple[float, float, float, float]) -> tuple[float, float, float, float]:
        measurement = xyxy_to_cxcywh(np.asarray(bbox, dtype=np.float64)).reshape(4, 1)
        innovation = measurement - self.h @ self.x
        innovation_cov = self.h @ self.p @ self.h.T + self.r
        gain = self.p @ self.h.T @ np.linalg.inv(innovation_cov)
        self.x = self.x + gain @ innovation
        self.p = (np.eye(8, dtype=np.float64) - gain @ self.h) @ self.p
        self._clamp_size()
        return self.bbox()

    def bbox(self) -> tuple[float, float, float, float]:
        corners = cxcywh_to_xyxy(self.x[:4, 0])
        return (float(corners[0]), float(corners[1]), float(corners[2]), float(corners[3]))

    def _clamp_size(self) -> None:
        self.x[2, 0] = max(float(self.x[2, 0]), 1.0)
        self.x[3, 0] = max(float(self.x[3, 0]), 1.0)


class _Track:
    def __init__(self, track_id: int, detection: Detection) -> None:
        self.track_id = track_id
        self.filter = _KalmanBox(detection.bbox)
        self.bbox = detection.bbox
        self.score = float(detection.score)
        self.class_id = int(detection.class_id)
        self.class_name = detection.class_name
        self.hits = 1
        self.age = 1
        self.time_since_update = 0

    def predict(self) -> None:
        self.bbox = self.filter.predict()
        self.age += 1
        self.time_since_update += 1

    def update(self, detection: Detection) -> None:
        self.bbox = self.filter.update(detection.bbox)
        self.score = float(detection.score)
        self.class_name = detection.class_name
        self.hits += 1
        self.time_since_update = 0

    def snapshot(self, min_hits: int) -> TrackState:
        return TrackState(
            track_id=self.track_id,
            bbox=self.bbox,
            score=self.score,
            class_id=self.class_id,
            class_name=self.class_name,
            hits=self.hits,
            age=self.age,
            time_since_update=self.time_since_update,
            confirmed=self.hits >= min_hits,
        )
