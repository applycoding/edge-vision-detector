from __future__ import annotations

import logging
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Protocol

import numpy as np
import onnxruntime as ort

from edge_vision.labels import COCO_CLASSES
from edge_vision.nms import nms
from edge_vision.postprocess import Layout, decode_predictions
from edge_vision.preprocess import LetterboxMeta, preprocess, scale_boxes_to_original
from edge_vision.providers import resolve_providers
from edge_vision.types import Detection, StageTiming

logger = logging.getLogger(__name__)


class InferenceSession(Protocol):
    def get_inputs(self) -> Sequence[object]: ...

    def get_providers(self) -> Sequence[str]: ...

    def run(
        self, output_names: list[str] | None, input_feed: dict[str, np.ndarray]
    ) -> list[np.ndarray]: ...


class OnnxDetector:
    """YOLOv5/v8 ONNX detector. Providers are whatever this onnxruntime build exposes."""

    def __init__(
        self,
        model_path: str,
        input_size: tuple[int, int],
        conf_threshold: float,
        iou_threshold: float,
        providers: Sequence[str],
        class_names: Sequence[str] | None = None,
        layout: Layout = "auto",
        max_det: int = 100,
        session: InferenceSession | None = None,
    ) -> None:
        self.model_path = model_path
        self.input_size = (int(input_size[0]), int(input_size[1]))
        self.conf_threshold = float(conf_threshold)
        self.iou_threshold = float(iou_threshold)
        self.layout: Layout = layout
        self.max_det = int(max_det)
        self.class_names = tuple(class_names) if class_names is not None else COCO_CLASSES
        if session is None:
            self.session = _open_session(model_path, providers)
        else:
            self.session = session
        model_input = self.session.get_inputs()[0]
        self.input_name = str(getattr(model_input, "name"))
        self.providers = [str(name) for name in self.session.get_providers()]
        _check_static_input(getattr(model_input, "shape", ()), self.input_size)
        self._logged_output = False
        logger.info("ONNX Runtime providers: %s", ", ".join(self.providers))

    def detect(self, frame: np.ndarray) -> tuple[list[Detection], StageTiming]:
        started = time.perf_counter()
        blob, meta = preprocess(frame, self.input_size)
        preprocessed = time.perf_counter()
        outputs = self.session.run(None, {self.input_name: blob})
        inferred = time.perf_counter()
        if not self._logged_output:
            logger.info("model output shape %s", tuple(np.shape(outputs[0])))
            self._logged_output = True
        detections = self._decode(outputs[0], meta)
        finished = time.perf_counter()
        timing = StageTiming(
            preprocess_s=preprocessed - started,
            inference_s=inferred - preprocessed,
            postprocess_s=finished - inferred,
        )
        return detections, timing

    def _decode(self, output: np.ndarray, meta: LetterboxMeta) -> list[Detection]:
        boxes, scores, class_ids = decode_predictions(
            output,
            self.conf_threshold,
            layout=self.layout,
            num_classes=len(self.class_names),
        )
        keep = nms(boxes, scores, self.iou_threshold, class_ids)
        if self.max_det > 0:
            keep = keep[: self.max_det]
        boxes = boxes[keep]
        scores = scores[keep]
        class_ids = class_ids[keep]
        boxes = scale_boxes_to_original(boxes, meta)
        detections: list[Detection] = []
        for box, score, class_id in zip(boxes, scores, class_ids, strict=True):
            if box[2] <= box[0] or box[3] <= box[1]:
                continue
            index = int(class_id)
            name = self.class_names[index] if 0 <= index < len(self.class_names) else f"class_{index}"
            detections.append(
                Detection(
                    bbox=(float(box[0]), float(box[1]), float(box[2]), float(box[3])),
                    score=float(score),
                    class_id=index,
                    class_name=name,
                )
            )
        return detections


def _open_session(model_path: str, providers: Sequence[str]) -> ort.InferenceSession:
    path = Path(model_path)
    if not path.is_file():
        raise FileNotFoundError(
            f"model not found: {path}. Download one with `python scripts/download_model.py`."
        )
    available = ort.get_available_providers()
    selected = resolve_providers(tuple(providers), available)
    missing = [name for name in providers if name not in set(available)]
    if missing:
        logger.warning("requested providers are not installed: %s", ", ".join(missing))
    logger.info("creating session with providers: %s", ", ".join(selected))
    return ort.InferenceSession(str(path), providers=selected)


def _check_static_input(shape: object, input_size: tuple[int, int]) -> None:
    if not isinstance(shape, (list, tuple)) or len(shape) != 4:
        return
    height = _fixed_dim(shape[2])
    width = _fixed_dim(shape[3])
    if height is None or width is None:
        return
    if (height, width) != input_size:
        raise ValueError(
            f"model input is {height}x{width} (height, width) but config input_size is "
            f"{input_size[0]}x{input_size[1]}"
        )


def _fixed_dim(value: object) -> int | None:
    if isinstance(value, bool) or isinstance(value, str):
        return None
    if isinstance(value, (int, np.integer)):
        return int(value)
    return None
