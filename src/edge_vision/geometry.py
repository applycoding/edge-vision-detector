from __future__ import annotations

import numpy as np


def xywh_to_xyxy(boxes: np.ndarray) -> np.ndarray:
    """Convert center-x, center-y, width, height boxes to corner boxes."""
    boxes = np.asarray(boxes, dtype=np.float32)
    out = np.empty_like(boxes)
    out[:, 0] = boxes[:, 0] - boxes[:, 2] / 2.0
    out[:, 1] = boxes[:, 1] - boxes[:, 3] / 2.0
    out[:, 2] = boxes[:, 0] + boxes[:, 2] / 2.0
    out[:, 3] = boxes[:, 1] + boxes[:, 3] / 2.0
    return out


def xyxy_to_cxcywh(boxes: np.ndarray) -> np.ndarray:
    """Convert corner boxes to center-x, center-y, width, height."""
    array = np.asarray(boxes, dtype=np.float64)
    single = array.ndim == 1
    if single:
        array = array.reshape(1, 4)
    out = np.empty_like(array)
    out[:, 0] = (array[:, 0] + array[:, 2]) / 2.0
    out[:, 1] = (array[:, 1] + array[:, 3]) / 2.0
    out[:, 2] = array[:, 2] - array[:, 0]
    out[:, 3] = array[:, 3] - array[:, 1]
    if single:
        return out[0]
    return out


def cxcywh_to_xyxy(boxes: np.ndarray) -> np.ndarray:
    """Convert center boxes back to corners. Accepts one box or a batch."""
    array = np.asarray(boxes, dtype=np.float64)
    single = array.ndim == 1
    if single:
        array = array.reshape(1, 4)
    out = np.empty_like(array)
    out[:, 0] = array[:, 0] - array[:, 2] / 2.0
    out[:, 1] = array[:, 1] - array[:, 3] / 2.0
    out[:, 2] = array[:, 0] + array[:, 2] / 2.0
    out[:, 3] = array[:, 1] + array[:, 3] / 2.0
    if single:
        return out[0]
    return out


def box_iou_matrix(boxes_a: np.ndarray, boxes_b: np.ndarray) -> np.ndarray:
    """Pairwise IoU for xyxy boxes. Shape (N, M)."""
    left = np.asarray(boxes_a, dtype=np.float32)
    right = np.asarray(boxes_b, dtype=np.float32)
    if left.size == 0 or right.size == 0:
        return np.zeros((left.shape[0], right.shape[0]), dtype=np.float32)
    top_left = np.maximum(left[:, None, :2], right[None, :, :2])
    bottom_right = np.minimum(left[:, None, 2:], right[None, :, 2:])
    wh = np.clip(bottom_right - top_left, 0, None)
    intersection = wh[:, :, 0] * wh[:, :, 1]
    area_left = np.clip(left[:, 2] - left[:, 0], 0, None) * np.clip(left[:, 3] - left[:, 1], 0, None)
    area_right = np.clip(right[:, 2] - right[:, 0], 0, None) * np.clip(
        right[:, 3] - right[:, 1], 0, None
    )
    union = area_left[:, None] + area_right[None, :] - intersection
    iou = np.zeros_like(intersection, dtype=np.float32)
    np.divide(intersection, union, out=iou, where=union > 0)
    return iou
