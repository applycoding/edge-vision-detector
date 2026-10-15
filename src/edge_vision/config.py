from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import yaml

from edge_vision.postprocess import Layout
from edge_vision.zones import Zone

SourceKind = Literal["file", "camera", "rtsp"]
PublishMode = Literal["none", "http", "mqtt"]


@dataclass
class SourceConfig:
    kind: SourceKind
    path: str = ""
    device: int = 0
    url: str = ""


@dataclass
class ModelConfig:
    path: str = "models/yolov8n.onnx"
    input_size: tuple[int, int] = (640, 640)
    conf_threshold: float = 0.25
    iou_threshold: float = 0.45
    providers: tuple[str, ...] = ("CPUExecutionProvider",)
    class_names: tuple[str, ...] | None = None
    layout: Layout = "auto"
    max_detections: int = 100


@dataclass
class TrackerConfig:
    iou_threshold: float = 0.3
    max_age: int = 30
    min_hits: int = 3


@dataclass
class PublishConfig:
    mode: PublishMode = "none"
    http_url: str = "http://127.0.0.1:8080/events"
    http_timeout_s: float = 2.0
    mqtt_host: str = "127.0.0.1"
    mqtt_port: int = 1883
    mqtt_topic: str = "edge/alerts"
    mqtt_qos: int = 0


@dataclass
class OutputConfig:
    annotated_path: str = "output/annotated.mp4"
    show: bool = False
    metrics_every: int = 30


@dataclass
class AppConfig:
    source: SourceConfig
    model: ModelConfig
    tracker: TrackerConfig
    zones: tuple[Zone, ...]
    publish: PublishConfig
    output: OutputConfig
    max_frames: int | None = None


def load_config(path: str | Path) -> AppConfig:
    config_path = Path(path)
    try:
        text = config_path.read_text(encoding="utf-8")
    except OSError as exc:
        raise FileNotFoundError(f"config not found: {config_path}") from exc
    loaded = yaml.safe_load(text)
    if loaded is None:
        loaded = {}
    if not isinstance(loaded, dict):
        raise ValueError("config root must be a mapping")
    data = dict(loaded)
    _reject_unknown(data, {"source", "model", "tracker", "zones", "publish", "output", "max_frames"}, "config")
    if "source" not in data:
        raise ValueError("config is missing source")
    return AppConfig(
        source=_parse_source(data.get("source")),
        model=_parse_model(data.get("model", {})),
        tracker=_parse_tracker(data.get("tracker", {})),
        zones=_parse_zones(data.get("zones", [])),
        publish=_parse_publish(data.get("publish", {})),
        output=_parse_output(data.get("output", {})),
        max_frames=_optional_positive_int(data.get("max_frames"), "max_frames"),
    )


def _parse_source(value: object) -> SourceConfig:
    raw = _mapping(value, "source")
    _reject_unknown(raw, {"kind", "path", "device", "url"}, "source")
    kind = _source_kind(raw.get("kind"))
    path = str(raw.get("path", ""))
    device = _int_value(raw.get("device", 0), "source.device")
    url = str(raw.get("url", ""))
    if kind == "file" and not path:
        raise ValueError("source.path is required for a file source")
    if kind == "camera" and device < 0:
        raise ValueError("source.device must be >= 0")
    if kind == "rtsp" and not url:
        raise ValueError("source.url is required for an rtsp source")
    return SourceConfig(kind=kind, path=path, device=device, url=url)


def _parse_model(value: object) -> ModelConfig:
    raw = _mapping(value, "model")
    _reject_unknown(
        raw,
        {
            "path",
            "input_size",
            "conf_threshold",
            "iou_threshold",
            "providers",
            "class_names",
            "layout",
            "max_detections",
        },
        "model",
    )
    size = raw.get("input_size", [640, 640])
    if not isinstance(size, (list, tuple)) or len(size) != 2:
        raise ValueError("model.input_size must be [height, width]")
    height = _int_value(size[0], "model.input_size height")
    width = _int_value(size[1], "model.input_size width")
    if height <= 0 or width <= 0:
        raise ValueError("model.input_size must be positive")
    names = raw.get("class_names")
    class_names: tuple[str, ...] | None
    if names is None:
        class_names = None
    elif isinstance(names, list) and all(isinstance(item, str) for item in names):
        class_names = tuple(names)
    else:
        raise ValueError("model.class_names must be a list of strings")
    providers = raw.get("providers", ["CPUExecutionProvider"])
    if not isinstance(providers, list) or not providers or not all(isinstance(item, str) for item in providers):
        raise ValueError("model.providers must be a non-empty list of provider names")
    return ModelConfig(
        path=str(raw.get("path", "models/yolov8n.onnx")),
        input_size=(height, width),
        conf_threshold=_unit_float(raw.get("conf_threshold", 0.25), "model.conf_threshold"),
        iou_threshold=_unit_float(raw.get("iou_threshold", 0.45), "model.iou_threshold"),
        providers=tuple(providers),
        class_names=class_names,
        layout=_layout(raw.get("layout", "auto")),
        max_detections=_int_value(raw.get("max_detections", 100), "model.max_detections"),
    )


def _parse_tracker(value: object) -> TrackerConfig:
    raw = _mapping(value, "tracker")
    _reject_unknown(raw, {"iou_threshold", "max_age", "min_hits"}, "tracker")
    return TrackerConfig(
        iou_threshold=_unit_float(raw.get("iou_threshold", 0.3), "tracker.iou_threshold"),
        max_age=_int_value(raw.get("max_age", 30), "tracker.max_age"),
        min_hits=_int_value(raw.get("min_hits", 3), "tracker.min_hits"),
    )


def _parse_zones(value: object) -> tuple[Zone, ...]:
    if value is None:
        return ()
    if not isinstance(value, list):
        raise ValueError("zones must be a list")
    zones: list[Zone] = []
    seen: set[str] = set()
    for index, item in enumerate(value):
        raw = _mapping(item, f"zones[{index}]")
        _reject_unknown(raw, {"name", "polygon", "classes"}, f"zones[{index}]")
        name = raw.get("name")
        if not isinstance(name, str) or not name:
            raise ValueError(f"zones[{index}].name must be a non-empty string")
        if name in seen:
            raise ValueError(f"duplicate zone name {name}")
        seen.add(name)
        zones.append(
            Zone(
                name=name,
                polygon=_polygon(raw.get("polygon"), f"zones[{index}].polygon"),
                classes=_classes(raw.get("classes", []), f"zones[{index}].classes"),
            )
        )
    return tuple(zones)


def _parse_publish(value: object) -> PublishConfig:
    raw = _mapping(value, "publish")
    _reject_unknown(raw, {"mode", "http", "mqtt"}, "publish")
    http = _mapping(raw.get("http", {}), "publish.http")
    mqtt = _mapping(raw.get("mqtt", {}), "publish.mqtt")
    _reject_unknown(http, {"url", "timeout_s"}, "publish.http")
    _reject_unknown(mqtt, {"host", "port", "topic", "qos"}, "publish.mqtt")
    return PublishConfig(
        mode=_publish_mode(raw.get("mode", "none")),
        http_url=str(http.get("url", "http://127.0.0.1:8080/events")),
        http_timeout_s=_positive_float(http.get("timeout_s", 2.0), "publish.http.timeout_s"),
        mqtt_host=str(mqtt.get("host", "127.0.0.1")),
        mqtt_port=_int_value(mqtt.get("port", 1883), "publish.mqtt.port"),
        mqtt_topic=str(mqtt.get("topic", "edge/alerts")),
        mqtt_qos=_int_value(mqtt.get("qos", 0), "publish.mqtt.qos"),
    )


def _parse_output(value: object) -> OutputConfig:
    raw = _mapping(value, "output")
    _reject_unknown(raw, {"annotated_path", "show", "metrics_every"}, "output")
    annotated = raw.get("annotated_path", "output/annotated.mp4")
    if annotated is None:
        annotated = ""
    if not isinstance(annotated, str):
        raise ValueError("output.annotated_path must be a string or null")
    show = raw.get("show", False)
    if not isinstance(show, bool):
        raise ValueError("output.show must be a boolean")
    return OutputConfig(
        annotated_path=annotated,
        show=show,
        metrics_every=_int_value(raw.get("metrics_every", 30), "output.metrics_every"),
    )


def _polygon(value: object, label: str) -> tuple[tuple[float, float], ...]:
    if not isinstance(value, list) or len(value) < 3:
        raise ValueError(f"{label} must have at least 3 points")
    points: list[tuple[float, float]] = []
    for index, point in enumerate(value):
        if not isinstance(point, (list, tuple)) or len(point) != 2:
            raise ValueError(f"{label} point {index} must be [x, y]")
        points.append((_number(point[0], f"{label}[{index}].x"), _number(point[1], f"{label}[{index}].y")))
    return tuple(points)


def _classes(value: object, label: str) -> tuple[int | str, ...]:
    if value is None:
        return ()
    if not isinstance(value, list):
        raise ValueError(f"{label} must be a list")
    items: list[int | str] = []
    for index, item in enumerate(value):
        if isinstance(item, bool) or not isinstance(item, (int, str)):
            raise ValueError(f"{label}[{index}] must be a class name or class id")
        items.append(item)
    return tuple(items)


def _mapping(value: object, label: str) -> dict[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be a mapping")
    return dict(value)


def _reject_unknown(mapping: dict[str, Any], allowed: set[str], label: str) -> None:
    unknown = sorted(set(mapping) - allowed)
    if unknown:
        raise ValueError(f"unknown {label} fields: {', '.join(unknown)}")


def _source_kind(value: object) -> SourceKind:
    if value == "file":
        return "file"
    if value == "camera":
        return "camera"
    if value == "rtsp":
        return "rtsp"
    raise ValueError(f"source.kind must be file, camera, or rtsp (got {value!r})")


def _publish_mode(value: object) -> PublishMode:
    if value == "none":
        return "none"
    if value == "http":
        return "http"
    if value == "mqtt":
        return "mqtt"
    raise ValueError(f"publish.mode must be none, http, or mqtt (got {value!r})")


def _layout(value: object) -> Layout:
    if value == "auto":
        return "auto"
    if value == "yolov8":
        return "yolov8"
    if value == "yolov5":
        return "yolov5"
    if value == "end2end":
        return "end2end"
    raise ValueError(f"model.layout must be auto, yolov8, yolov5, or end2end (got {value!r})")


def _optional_positive_int(value: object, label: str) -> int | None:
    if value is None:
        return None
    parsed = _int_value(value, label)
    if parsed < 0:
        raise ValueError(f"{label} must be >= 0")
    return parsed


def _int_value(value: object, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{label} must be an integer")
    return value


def _number(value: object, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} must be a number")
    return float(value)


def _unit_float(value: object, label: str) -> float:
    parsed = _number(value, label)
    if not 0.0 < parsed <= 1.0:
        raise ValueError(f"{label} must be in (0, 1]")
    return parsed


def _positive_float(value: object, label: str) -> float:
    parsed = _number(value, label)
    if parsed <= 0:
        raise ValueError(f"{label} must be > 0")
    return parsed
