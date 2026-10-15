"""Export an Ultralytics YOLO checkpoint to ONNX.

This is optional and is not used by the tests. Install the extra first:

    pip install -e ".[export]"

Then:

    python scripts/export_model.py --weights yolov8n.pt --output models/yolov8n.onnx
"""

from __future__ import annotations

import argparse
from pathlib import Path

from ultralytics import YOLO


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Export a YOLO .pt checkpoint to ONNX")
    parser.add_argument("--weights", default="yolov8n.pt")
    parser.add_argument("--output", default="models/yolov8n.onnx")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--opset", type=int, default=12)
    args = parser.parse_args(argv)
    model = YOLO(args.weights)
    exported = Path(model.export(format="onnx", imgsz=args.imgsz, opset=args.opset, simplify=True))
    destination = Path(args.output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if exported.resolve() != destination.resolve():
        destination.write_bytes(exported.read_bytes())
    print(destination)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
