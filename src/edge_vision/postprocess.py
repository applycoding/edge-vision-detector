from __future__ import annotations

from typing import Literal

import numpy as np

from edge_vision.geometry import xywh_to_xyxy

Layout = Literal["auto", "yolov8", "yolov5", "end2end"]


def decode_predictions(
    output: np.ndarray,
    conf_threshold: float,
    *,
    layout: Layout = "auto",
    num_classes: int | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Decode a YOLO head into xyxy boxes, scores, and class ids.

    `auto` understands three exports:

    - YOLOv8: (N, 4 + classes) or (4 + classes, N), no objectness
    - YOLOv5: (N, 5 + classes) with objectness in column 4
    - end-to-end NMS: (N, 6) as x1, y1, x2, y2, score, class

    Scores are treated as probabilities, which is what an Ultralytics ONNX
    export emits. Boxes are in the network input's pixel coordinate system.
    """
    rows = _as_detection_rows(output, num_classes=num_classes, layout=layout)
    chosen = layout if layout != "auto" else _infer_layout(rows.shape[1], num_classes)
    if chosen == "end2end":
        return _decode_end2end(rows, conf_threshold)
    if chosen == "yolov5":
        return _decode_yolov5(rows, conf_threshold)
    if chosen == "yolov8":
        return _decode_yolov8(rows, conf_threshold)
    _never: never = chosen
    raise ValueError(f"unsupported layout {_never}")


def _as_detection_rows(output: np.ndarray, *, num_classes: int | None, layout: Layout) -> np.ndarray:
    pred = np.asarray(output, dtype=np.float32)
    pred = np.squeeze(pred)
    if pred.ndim == 1:
        pred = pred.reshape(1, -1)
    if pred.ndim != 2:
        raise ValueError(
            f"expected a prediction tensor that squeezes to 2D, got shape {tuple(np.shape(output))}"
        )
    widths = _candidate_widths(num_classes, layout)
    if pred.shape[1] in widths:
        return np.ascontiguousarray(pred)
    if pred.shape[0] in widths:
        return np.ascontiguousarray(pred.T)
    if pred.shape[0] < pred.shape[1]:
        return np.ascontiguousarray(pred.T)
    return np.ascontiguousarray(pred)


def _candidate_widths(num_classes: int | None, layout: Layout) -> set[int]:
    if layout == "end2end":
        return {6}
    if layout == "yolov8" and num_classes is not None:
        return {4 + num_classes}
    if layout == "yolov5" and num_classes is not None:
        return {5 + num_classes}
    widths = {6, 84, 85}
    if num_classes is not None:
        widths.add(4 + num_classes)
        widths.add(5 + num_classes)
    return widths


def _infer_layout(width: int, num_classes: int | None) -> Layout:
    # Width 6 is the end-to-end row. A 2-class YOLOv8 head is also width 6;
    # set model.layout to yolov8 if that is the export you have.
    if width == 6:
        return "end2end"
    if num_classes is not None and width == 4 + num_classes:
        return "yolov8"
    if num_classes is not None and width == 5 + num_classes:
        return "yolov5"
    if width == 85:
        return "yolov5"
    return "yolov8"


def _decode_yolov8(rows: np.ndarray, conf_threshold: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if rows.shape[1] < 5:
        raise ValueError(f"YOLOv8 head width must be at least 5, got {rows.shape[1]}")
    class_scores = rows[:, 4:]
    class_ids = np.argmax(class_scores, axis=1)
    scores = class_scores[np.arange(class_scores.shape[0]), class_ids]
    return _filter_xywh(rows[:, :4], scores, class_ids, conf_threshold)


def _decode_yolov5(rows: np.ndarray, conf_threshold: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if rows.shape[1] < 6:
        raise ValueError(f"YOLOv5 head width must be at least 6, got {rows.shape[1]}")
    objectness = rows[:, 4]
    class_scores = rows[:, 5:]
    class_ids = np.argmax(class_scores, axis=1)
    scores = objectness * class_scores[np.arange(class_scores.shape[0]), class_ids]
    return _filter_xywh(rows[:, :4], scores, class_ids, conf_threshold)


def _decode_end2end(rows: np.ndarray, conf_threshold: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if rows.shape[1] < 6:
        raise ValueError(f"end-to-end rows must have at least 6 columns, got {rows.shape[1]}")
    scores = rows[:, 4]
    keep = scores >= conf_threshold
    if not np.any(keep):
        return _empty()
    boxes = rows[keep, :4].astype(np.float32, copy=False)
    class_ids = rows[keep, 5].astype(np.int64)
    return boxes, scores[keep].astype(np.float32, copy=False), class_ids


def _filter_xywh(
    boxes_xywh: np.ndarray,
    scores: np.ndarray,
    class_ids: np.ndarray,
    conf_threshold: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    keep = scores >= conf_threshold
    if not np.any(keep):
        return _empty()
    return (
        xywh_to_xyxy(boxes_xywh[keep]),
        scores[keep].astype(np.float32, copy=False),
        class_ids[keep].astype(np.int64),
    )


def _empty() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    return (
        np.zeros((0, 4), dtype=np.float32),
        np.zeros((0,), dtype=np.float32),
        np.zeros((0,), dtype=np.int64),
    )
