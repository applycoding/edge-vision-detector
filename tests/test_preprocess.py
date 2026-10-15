import numpy as np
import pytest

from edge_vision.preprocess import letterbox, preprocess, scale_boxes_to_input, scale_boxes_to_original


def test_letterbox_pads_and_keeps_aspect() -> None:
    image = np.zeros((10, 10, 3), dtype=np.uint8)
    padded, meta = letterbox(image, (30, 20))
    assert padded.shape == (30, 20, 3)
    assert meta.pad_x == 0
    assert meta.pad_y == 5
    assert meta.scale_x == pytest.approx(2.0)
    assert meta.scale_y == pytest.approx(2.0)
    assert int(padded[0, 0, 0]) == 114
    assert int(padded[5, 0, 0]) == 0


def test_letterbox_wide_frame() -> None:
    image = np.zeros((100, 200, 3), dtype=np.uint8)
    padded, meta = letterbox(image, (50, 50))
    assert padded.shape == (50, 50, 3)
    assert meta.pad_x == 0
    assert meta.pad_y == 12
    assert meta.scale_x == pytest.approx(0.25)
    assert meta.scale_y == pytest.approx(0.25)


def test_scale_boxes_roundtrip() -> None:
    image = np.zeros((10, 10, 3), dtype=np.uint8)
    _padded, meta = letterbox(image, (30, 20))
    original = np.array([[1, 1, 4, 5]], dtype=np.float32)
    projected = scale_boxes_to_input(original, meta)
    restored = scale_boxes_to_original(projected, meta)
    np.testing.assert_allclose(restored, original, atol=1e-5)


def test_scale_boxes_clips_to_the_frame() -> None:
    image = np.zeros((10, 10, 3), dtype=np.uint8)
    _padded, meta = letterbox(image, (10, 10))
    boxes = np.array([[-50, -50, 80, 80]], dtype=np.float32)
    restored = scale_boxes_to_original(boxes, meta)
    assert restored[0].min() >= 0
    assert restored[0, 2] <= 10
    assert restored[0, 3] <= 10


def test_preprocess_blob_is_rgb_and_normalized() -> None:
    image = np.zeros((8, 8, 3), dtype=np.uint8)
    image[:, :] = (255, 0, 0)  # BGR blue
    blob, meta = preprocess(image, (8, 8))
    assert blob.shape == (1, 3, 8, 8)
    assert blob.dtype == np.float32
    assert blob.flags.c_contiguous
    assert meta.pad_x == 0 and meta.pad_y == 0
    assert float(blob.min()) >= 0.0
    assert float(blob.max()) <= 1.0
    np.testing.assert_allclose(blob[0, 0], 0.0)
    np.testing.assert_allclose(blob[0, 2], 1.0)


def test_preprocess_rejects_bad_frames() -> None:
    with pytest.raises(ValueError):
        preprocess(np.zeros((8, 8), dtype=np.uint8), (8, 8))
    with pytest.raises(ValueError):
        preprocess(np.zeros((8, 8, 3), dtype=np.float32), (8, 8))
