from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Detection:
    """One detector box in the original frame, xyxy pixels."""

    bbox: tuple[float, float, float, float]
    score: float
    class_id: int
    class_name: str


@dataclass(frozen=True)
class TrackState:
    """Confirmed or tentative track exposed to zones and the annotator."""

    track_id: int
    bbox: tuple[float, float, float, float]
    score: float
    class_id: int
    class_name: str
    hits: int
    age: int
    time_since_update: int
    confirmed: bool


@dataclass(frozen=True)
class StageTiming:
    preprocess_s: float
    inference_s: float
    postprocess_s: float
