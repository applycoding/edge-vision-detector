import pytest

from edge_vision.providers import resolve_providers


def test_keeps_requested_order_and_appends_cpu() -> None:
    available = ["TensorrtExecutionProvider", "CUDAExecutionProvider", "CPUExecutionProvider"]
    assert resolve_providers(["CUDAExecutionProvider"], available) == [
        "CUDAExecutionProvider",
        "CPUExecutionProvider",
    ]
    assert resolve_providers(["CPUExecutionProvider"], available) == ["CPUExecutionProvider"]


def test_falls_back_to_cpu_when_nothing_requested_is_installed() -> None:
    assert resolve_providers(
        ["TensorrtExecutionProvider", "CUDAExecutionProvider"],
        ["CPUExecutionProvider"],
    ) == ["CPUExecutionProvider"]


def test_errors_when_no_provider_can_run() -> None:
    with pytest.raises(RuntimeError):
        resolve_providers(["TensorrtExecutionProvider"], [])
    with pytest.raises(ValueError):
        resolve_providers([], ["CPUExecutionProvider"])
