from __future__ import annotations

import numpy as np

from edge_vision.geometry import box_iou_matrix


def nms(
    boxes: np.ndarray,
    scores: np.ndarray,
    iou_threshold: float,
    class_ids: np.ndarray | None = None,
) -> np.ndarray:
    """Greedy class-aware NMS. Suppresses a box when IoU is strictly greater than the threshold.

    Returns indices into `boxes`, highest score first. Equal scores keep the earlier index.
    """
    box_array = np.asarray(boxes, dtype=np.float32)
    score_array = np.asarray(scores, dtype=np.float32)
    if box_array.size == 0:
        return np.empty(0, dtype=np.int64)
    if box_array.ndim != 2 or box_array.shape[1] != 4:
        raise ValueError("boxes must have shape (N, 4)")
    if score_array.shape != (box_array.shape[0],):
        raise ValueError("scores must have shape (N,)")
    if class_ids is None:
        kept = _nms_one_class(box_array, score_array, iou_threshold)
        return kept
    labels = np.asarray(class_ids)
    if labels.shape != (box_array.shape[0],):
        raise ValueError("class_ids must have shape (N,)")
    kept_indices: list[int] = []
    for label in np.unique(labels):
        subset = np.flatnonzero(labels == label)
        local = _nms_one_class(box_array[subset], score_array[subset], iou_threshold)
        kept_indices.extend(int(subset[index]) for index in local)
    kept = np.asarray(kept_indices, dtype=np.int64)
    order = np.argsort(-score_array[kept], kind="mergesort")
    return kept[order]


def _nms_one_class(boxes: np.ndarray, scores: np.ndarray, iou_threshold: float) -> np.ndarray:
    order = np.argsort(-scores, kind="mergesort")
    kept: list[int] = []
    while order.size > 0:
        current = int(order[0])
        kept.append(current)
        if order.size == 1:
            break
        rest = order[1:]
        ious = box_iou_matrix(boxes[current : current + 1], boxes[rest])[0]
        order = rest[ious <= iou_threshold]
    return np.asarray(kept, dtype=np.int64)
