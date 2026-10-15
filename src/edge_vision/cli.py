from __future__ import annotations

import argparse
import json
import logging
import os
import platform
from pathlib import Path

import cv2
import onnxruntime as ort

from edge_vision import __version__
from edge_vision.capture import open_source
from edge_vision.config import AppConfig, load_config
from edge_vision.detector import OnnxDetector
from edge_vision.metrics import format_report
from edge_vision.pipeline import Pipeline
from edge_vision.publish import build_publisher
from edge_vision.tracker import SortTracker
from edge_vision.zones import ZoneMonitor

logger = logging.getLogger("edge_vision")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="edge-vision",
        description="Real-time ONNX detection, tracking, and zone alerts for edge devices.",
    )
    parser.add_argument("--version", action="version", version=f"edge-vision {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)

    run = sub.add_parser("run", help="Run the pipeline and write annotated video")
    _add_common(run)
    run.add_argument("--max-frames", type=int, default=None)
    run.add_argument("--warmup", type=int, default=0)
    run.add_argument("--publish", choices=["none", "http", "mqtt"], default=None)

    bench = sub.add_parser("bench", help="Time the pipeline and print JSON metrics")
    _add_common(bench)
    bench.add_argument("--frames", type=int, default=40)
    bench.add_argument("--warmup", type=int, default=10)
    return parser


def _add_common(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--config", required=True, help="YAML config path")
    parser.add_argument("--model", help="Override the ONNX model path")
    parser.add_argument("--source", help="Override the file source path")
    parser.add_argument(
        "--provider",
        action="append",
        dest="providers",
        help="ONNX Runtime provider. Repeat to set the preference order.",
    )


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    command = args.command
    try:
        if command == "run":
            summary = _execute(args, bench=False)
            print(format_report(summary))
            return 0
        if command == "bench":
            summary = _execute(args, bench=True)
            print(json.dumps(summary, indent=2))
            return 0
    except (FileNotFoundError, RuntimeError, ValueError, OSError) as exc:
        logger.error("%s", exc)
        return 1
    raise RuntimeError(f"unknown command {command}")


def cli_entry() -> None:
    raise SystemExit(main())


def _execute(args: argparse.Namespace, *, bench: bool) -> dict[str, object]:
    cfg = load_config(args.config)
    _apply_overrides(cfg, args)
    if bench:
        cfg.output.annotated_path = ""
        cfg.output.show = False
    _require_model(cfg.model.path)
    source = open_source(cfg.source)
    publisher = build_publisher(cfg.publish)
    try:
        detector = OnnxDetector(
            model_path=cfg.model.path,
            input_size=cfg.model.input_size,
            conf_threshold=cfg.model.conf_threshold,
            iou_threshold=cfg.model.iou_threshold,
            providers=cfg.model.providers,
            class_names=cfg.model.class_names,
            layout=cfg.model.layout,
            max_det=cfg.model.max_detections,
        )
        pipeline = Pipeline(
            source=source,
            detector=detector,
            tracker=SortTracker(
                iou_threshold=cfg.tracker.iou_threshold,
                max_age=cfg.tracker.max_age,
                min_hits=cfg.tracker.min_hits,
            ),
            zone_monitor=ZoneMonitor(cfg.zones),
            zones=cfg.zones,
            publisher=publisher,
            output_path=cfg.output.annotated_path,
            show=cfg.output.show,
            metrics_every=0 if bench else cfg.output.metrics_every,
        )
        measured = args.frames if bench else cfg.max_frames
        summary = pipeline.run(max_frames=measured, warmup=args.warmup)
    except Exception:
        source.close()
        publisher.close()
        raise
    if bench:
        summary["hardware"] = _hardware()
        summary["onnxruntime"] = ort.__version__
        summary["opencv"] = cv2.__version__
        summary["providers"] = list(detector.providers)
        summary["model"] = cfg.model.path
        summary["input_size"] = [cfg.model.input_size[0], cfg.model.input_size[1]]
        summary["warmup_frames"] = args.warmup
    return summary


def _apply_overrides(cfg: AppConfig, args: argparse.Namespace) -> None:
    if args.model:
        cfg.model.path = args.model
    if args.source:
        if cfg.source.kind != "file":
            raise ValueError("--source overrides a file source; edit the config for camera or rtsp")
        cfg.source.path = args.source
    if args.providers:
        cfg.model.providers = tuple(args.providers)
    publish = getattr(args, "publish", None)
    if publish is not None:
        cfg.publish.mode = publish
    max_frames = getattr(args, "max_frames", None)
    if max_frames is not None:
        if max_frames < 0:
            raise ValueError("--max-frames must be >= 0")
        cfg.max_frames = max_frames


def _require_model(path: str) -> None:
    if Path(path).is_file():
        return
    raise FileNotFoundError(
        f"model not found: {path}. Download one with `python scripts/download_model.py`."
    )


def _hardware() -> dict[str, object]:
    return {
        "platform": platform.platform(),
        "processor": _cpu_model(),
        "cpu_count": os.cpu_count(),
        "python": platform.python_version(),
    }


def _cpu_model() -> str:
    try:
        text = Path("/proc/cpuinfo").read_text(encoding="utf-8")
    except OSError:
        return platform.processor()
    for line in text.splitlines():
        if line.lower().startswith("model name"):
            return line.split(":", 1)[1].strip()
    return platform.processor()
