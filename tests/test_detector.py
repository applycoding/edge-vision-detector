import numpy as np
import pytest

from edge_vision.detector import OnnxDetector


class _Input:
    def __init__(self, name: str, shape: list[object]) -> None:
        self.name = name
        self.shape = shape


class _Session:
    def __init__(self) -> None:
        self.feeds: list[np.ndarray] = []

    def get_inputs(self) -> list[_Input]:
        return [_Input("images", [1, 3, 32, 32])]

    def get_providers(self) -> list[str]:
        return ["CPUExecutionProvider"]

    def run(self, _names: list[str] | None, feed: dict[str, np.ndarray]) -> list[np.ndarray]:
        blob = feed["images"]
        assert blob.shape == (1, 3, 32, 32)
        assert blob.dtype == np.float32
        self.feeds.append(blob)
        output = np.zeros((1, 84, 5), dtype=np.float32)
        output[0, 0, 0] = 16
        output[0, 1, 0] = 16
        output[0, 2, 0] = 8
        output[0, 3, 0] = 8
        output[0, 4, 0] = 0.92
        output[0, 0, 1] = 16.5
        output[0, 1, 1] = 16
        output[0, 2, 1] = 8
        output[0, 3, 1] = 8
        output[0, 4, 1] = 0.40
        output[0, 0, 2] = 8
        output[0, 1, 2] = 8
        output[0, 2, 2] = 4
        output[0, 3, 2] = 4
        output[0, 6, 2] = 0.55
        return [output]


def test_detector_decodes_a_fake_session_without_a_model_file() -> None:
    session = _Session()
    detector = OnnxDetector(
        model_path="unused.onnx",
        input_size=(32, 32),
        conf_threshold=0.25,
        iou_threshold=0.45,
        providers=["CPUExecutionProvider"],
        session=session,
    )
    frame = np.zeros((32, 32, 3), dtype=np.uint8)
    detections, timing = detector.detect(frame)
    assert detector.providers == ["CPUExecutionProvider"]
    assert timing.inference_s >= 0
    assert len(detections) == 2
    by_name = {item.class_name: item for item in detections}
    assert set(by_name) == {"person", "car"}
    np.testing.assert_allclose(by_name["person"].bbox, (12, 12, 20, 20), atol=1e-4)
    assert by_name["person"].score == pytest.approx(0.92)
