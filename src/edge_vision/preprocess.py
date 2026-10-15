from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

PAD_VALUE = 114


@dataclass(frozen=True)
class LetterboxMeta:
    """How `letterbox` mapped the original frame onto the network input."""

    scale_x: float
    scale_y: float
    pad_x: float
    pad_y: float
    original_size: tuple[int, int]
    input_size: tuple[int, int]


def preprocess(
    image: np.ndarray, input_size: tuple[int, int]
) -> tuple[np.ndarray, LetterboxMeta]:
    """Letterbox a uint8 BGR frame and return a 1x3xHxW float32 blob in RGB."""
    padded, meta = letterbox(image, input_size)
    rgb = cv2.cvtColor(padded, cv2.COLOR_BGR2RGB)
    blob = rgb.astype(np.float32) / 255.0
    blob = np.transpose(blob, (2, 0, 1))[None, ...]
    return np.ascontiguousarray(blob), meta


def letterbox(
    image: np.ndarray,
    input_size: tuple[int, int],
    color: tuple[int, int, int] = (PAD_VALUE, PAD_VALUE, PAD_VALUE),
) -> tuple[np.ndarray, LetterboxMeta]:
    """Resize `image` to fit inside `input_size` (height, width) and pad."""
    frame = _require_bgr(image)
    dst_h, dst_w = int(input_size[0]), int(input_size[1])
    if dst_h <= 0 or dst_w <= 0:
        raise ValueError("input_size must be positive")
    src_h, src_w = frame.shape[:2]
    scale = min(dst_w / src_w, dst_h / src_h)
    resized_w = min(dst_w, max(1, int(round(src_w * scale))))
    resized_h = min(dst_h, max(1, int(round(src_h * scale))))
    resized = cv2.resize(frame, (resized_w, resized_h), interpolation=cv2.INTER_LINEAR)
    pad_w = dst_w - resized_w
    pad_h = dst_h - resized_h
    left = pad_w // 2
    right = pad_w - left
    top = pad_h // 2
    bottom = pad_h - top
    padded = cv2.copyMakeBorder(
        resized, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color
    )
    meta = LetterboxMeta(
        scale_x=resized_w / src_w,
        scale_y=resized_h / src_h,
        pad_x=float(left),
        pad_y=float(top),
        original_size=(src_w, src_h),
        input_size=(dst_h, dst_w),
    )
    return padded, meta


def scale_boxes_to_original(boxes: np.ndarray, meta: LetterboxMeta) -> np.ndarray:
    """Map xyxy boxes from network-input pixels back to the original frame."""
    array = np.asarray(boxes, dtype=np.float32)
    if array.size == 0:
        return np.zeros((0, 4), dtype=np.float32)
    out = array.copy()
    out[:, [0, 2]] -= meta.pad_x
    out[:, [1, 3]] -= meta.pad_y
    out[:, [0, 2]] /= meta.scale_x
    out[:, [1, 3]] /= meta.scale_y
    width, height = meta.original_size
    out[:, [0, 2]] = np.clip(out[:, [0, 2]], 0, width)
    out[:, [1, 3]] = np.clip(out[:, [1, 3]], 0, height)
    return out


def scale_boxes_to_input(boxes: np.ndarray, meta: LetterboxMeta) -> np.ndarray:
    """Forward map used to check `scale_boxes_to_original`."""
    array = np.asarray(boxes, dtype=np.float32)
    out = array.copy()
    out[:, [0, 2]] = out[:, [0, 2]] * meta.scale_x + meta.pad_x
    out[:, [1, 3]] = out[:, [1, 3]] * meta.scale_y + meta.pad_y
    return out


def _require_bgr(image: np.ndarray) -> np.ndarray:
    if not isinstance(image, np.ndarray):
        raise TypeError("frame must be a numpy array")
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError(f"expected an HxWx3 BGR frame, got shape {getattr(image, 'shape', None)}")
    if image.dtype != np.uint8:
        raise ValueError("expected a uint8 BGR frame")
    if image.shape[0] == 0 or image.shape[1] == 0:
        raise ValueError("empty frame")
    return image
