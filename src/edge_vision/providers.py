from __future__ import annotations

from collections.abc import Sequence


def resolve_providers(requested: Sequence[str], available: Sequence[str]) -> list[str]:
    """Keep requested providers that this runtime actually has, CPU last as a fallback.

    The returned order is the order passed to `onnxruntime.InferenceSession`.
    """
    if not requested:
        raise ValueError("at least one ONNX Runtime provider is required")
    available_set = set(available)
    chosen = [name for name in requested if name in available_set]
    if not chosen:
        if "CPUExecutionProvider" not in available_set:
            joined = ", ".join(requested)
            raise RuntimeError(f"none of the requested ONNX Runtime providers are installed: {joined}")
        return ["CPUExecutionProvider"]
    if "CPUExecutionProvider" in available_set and "CPUExecutionProvider" not in chosen:
        chosen.append("CPUExecutionProvider")
    return chosen
