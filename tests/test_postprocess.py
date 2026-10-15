import numpy as np

from edge_vision.labels import COCO_CLASSES
from edge_vision.nms import nms
from edge_vision.postprocess import decode_predictions


def test_coco_names_match_the_ultralytics_head() -> None:
    assert len(COCO_CLASSES) == 80
    assert COCO_CLASSES[0] == "person"
    assert COCO_CLASSES[2] == "car"
    assert COCO_CLASSES[-1] == "toothbrush"


def test_decode_yolov8_channels_first() -> None:
    pred = np.zeros((1, 84, 4), dtype=np.float32)
    pred[0, 0, 0] = 50
    pred[0, 1, 0] = 40
    pred[0, 2, 0] = 20
    pred[0, 3, 0] = 10
    pred[0, 4, 0] = 0.90
    pred[0, 0, 1] = 10
    pred[0, 1, 1] = 10
    pred[0, 2, 1] = 4
    pred[0, 3, 1] = 4
    pred[0, 6, 1] = 0.20
    pred[0, 0, 2] = 80
    pred[0, 1, 2] = 80
    pred[0, 2, 2] = 8
    pred[0, 3, 2] = 8
    pred[0, 7, 2] = 0.60
    boxes, scores, class_ids = decode_predictions(pred, 0.25, num_classes=80)
    assert boxes.shape == (2, 4)
    np.testing.assert_allclose(boxes[0], [40, 35, 60, 45])
    np.testing.assert_allclose(scores, [0.90, 0.60])
    np.testing.assert_array_equal(class_ids, [0, 3])


def test_decode_then_nms_suppresses_the_same_class() -> None:
    pred = np.zeros((1, 84, 2), dtype=np.float32)
    pred[0, 0, 0] = 20
    pred[0, 1, 0] = 20
    pred[0, 2, 0] = 10
    pred[0, 3, 0] = 10
    pred[0, 4, 0] = 0.95
    pred[0, 0, 1] = 21
    pred[0, 1, 1] = 20
    pred[0, 2, 1] = 10
    pred[0, 3, 1] = 10
    pred[0, 4, 1] = 0.40
    boxes, scores, class_ids = decode_predictions(pred, 0.25, num_classes=80)
    keep = nms(boxes, scores, 0.45, class_ids)
    assert keep.tolist() == [0]
    np.testing.assert_allclose(scores[keep], [0.95])


def test_decode_yolov5_multiplies_objectness() -> None:
    pred = np.zeros((1, 7, 1), dtype=np.float32)
    pred[0, 0, 0] = 8
    pred[0, 1, 0] = 8
    pred[0, 2, 0] = 4
    pred[0, 3, 0] = 4
    pred[0, 4, 0] = 0.5
    pred[0, 6, 0] = 0.8
    boxes, scores, class_ids = decode_predictions(pred, 0.25, num_classes=2)
    assert len(boxes) == 1
    assert float(scores[0]) == np.float32(0.4)
    assert int(class_ids[0]) == 1
    empty_boxes, empty_scores, empty_ids = decode_predictions(pred, 0.5, num_classes=2)
    assert empty_boxes.shape == (0, 4)
    assert empty_scores.shape == (0,)
    assert empty_ids.shape == (0,)


def test_decode_end_to_end_rows() -> None:
    pred = np.array(
        [[[10, 12, 20, 30, 0.91, 0], [0, 0, 4, 4, 0.10, 1], [40, 40, 55, 70, 0.70, 3]]],
        dtype=np.float32,
    )
    boxes, scores, class_ids = decode_predictions(pred, 0.25, layout="end2end")
    assert boxes.shape == (2, 4)
    np.testing.assert_allclose(boxes[0], [10, 12, 20, 30])
    np.testing.assert_array_equal(class_ids, [0, 3])
    assert float(scores[1]) == np.float32(0.70)
