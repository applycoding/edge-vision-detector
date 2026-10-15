import numpy as np

from edge_vision.geometry import box_iou_matrix
from edge_vision.nms import nms


def test_empty_boxes() -> None:
    keep = nms(np.zeros((0, 4), dtype=np.float32), np.zeros((0,), dtype=np.float32), 0.5)
    assert keep.dtype == np.int64
    assert keep.tolist() == []


def test_identical_boxes_keep_the_higher_score() -> None:
    boxes = np.array([[0, 0, 10, 10], [0, 0, 10, 10], [50, 50, 60, 70]], dtype=np.float32)
    scores = np.array([0.40, 0.90, 0.50], dtype=np.float32)
    keep = nms(boxes, scores, 0.5)
    assert keep.tolist() == [1, 2]


def test_equal_scores_keep_the_earlier_index() -> None:
    boxes = np.array([[0, 0, 10, 10], [0, 0, 10, 10]], dtype=np.float32)
    scores = np.array([0.5, 0.5], dtype=np.float32)
    assert nms(boxes, scores, 0.5).tolist() == [0]


def test_iou_equal_to_the_threshold_is_kept() -> None:
    boxes = np.array([[0, 0, 10, 10], [10 / 3, 0, 10 / 3 + 10, 10]], dtype=np.float32)
    scores = np.array([0.9, 0.8], dtype=np.float32)
    iou = float(box_iou_matrix(boxes[:1], boxes[1:])[0, 0])
    assert iou == np.float32(0.5) or abs(iou - 0.5) < 1e-5
    assert len(nms(boxes, scores, iou)) == 2
    assert nms(boxes, scores, iou - 1e-3).tolist() == [0]


def test_different_classes_are_not_suppressed() -> None:
    boxes = np.array([[0, 0, 10, 10], [0, 0, 10, 10]], dtype=np.float32)
    scores = np.array([0.9, 0.8], dtype=np.float32)
    class_ids = np.array([0, 3])
    keep = nms(boxes, scores, 0.1, class_ids)
    assert set(keep.tolist()) == {0, 1}
