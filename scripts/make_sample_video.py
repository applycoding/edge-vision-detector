"""Write a short synthetic clip and a few PNG frames.

The rectangles are not photographs of COCO objects. The file is there so
capture, annotation, and timing can be exercised without a camera. A downloaded
detector will not reliably label these shapes.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np

WIDTH = 640
HEIGHT = 480
FPS = 20
FRAMES = 90


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Write samples/synthetic.mp4")
    parser.add_argument("--output", default="samples/synthetic.mp4")
    parser.add_argument("--frames", type=int, default=FRAMES)
    args = parser.parse_args(argv)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    frame_dir = output.parent / "frames"
    frame_dir.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(output),
        cv2.VideoWriter_fourcc(*"mp4v"),
        FPS,
        (WIDTH, HEIGHT),
    )
    if not writer.isOpened():
        raise RuntimeError(f"could not open {output}")
    snapshots = {0, args.frames // 3, (2 * args.frames) // 3}
    for index in range(args.frames):
        frame = render_frame(index)
        writer.write(frame)
        if index in snapshots:
            cv2.imwrite(str(frame_dir / f"frame_{index:03d}.png"), frame)
    writer.release()
    print(output)
    return 0


def render_frame(index: int) -> np.ndarray:
    frame = np.full((HEIGHT, WIDTH, 3), 28, np.uint8)
    frame[:, ::80] = 46
    frame[::80, :] = 46
    left = 40 + index * 3
    cv2.rectangle(frame, (left, 70), (left + 80, 210), (40, 40, 190), thickness=-1)
    right = 520 - index * 2
    cv2.rectangle(frame, (right, 250), (right + 100, 390), (180, 110, 30), thickness=-1)
    cv2.putText(
        frame,
        "synthetic",
        (16, HEIGHT - 24),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (230, 230, 230),
        1,
        cv2.LINE_AA,
    )
    return frame


if __name__ == "__main__":
    raise SystemExit(main())
