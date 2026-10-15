"""Download a small YOLOv8n ONNX model.

The weight file is not part of this repository. The default URL is the YOLOv8n
export listed in OpenCV's sample model catalog (sha1
68f864475d06e2ec4037181052739f268eeac38d). Ultralytics YOLOv8 weights are
licensed under AGPL-3.0, separate from this project's MIT code. Read that
license before shipping the file in a product.

Examples:
    python scripts/download_model.py
    python scripts/download_model.py --output models/yolov8n.onnx
"""

from __future__ import annotations

import argparse
import hashlib
import sys
import tempfile
import urllib.request
from pathlib import Path

PRIMARY_URL = "https://github.com/CVHub520/X-AnyLabeling/releases/download/v0.1.0/yolov8n.onnx"
PRIMARY_SHA1 = "68f864475d06e2ec4037181052739f268eeac38d"
MIRROR_URL = "https://huggingface.co/Kalray/yolov8/resolve/main/yolov8n.onnx"
USER_AGENT = "edge-vision-detector"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Download a YOLOv8n ONNX model")
    parser.add_argument("--output", default="models/yolov8n.onnx")
    parser.add_argument("--url", action="append", dest="urls", help="Override the download URL. Repeatable.")
    parser.add_argument("--sha1", default=None, help="Required sha1 when --url is set. Default URL checks a known sha1.")
    args = parser.parse_args(argv)
    destination = Path(args.output)
    candidates = _candidates(args.urls, args.sha1)
    if destination.is_file() and _matches_any(destination, candidates):
        print(f"already present: {destination} ({_sha1(destination)})")
        return 0
    destination.parent.mkdir(parents=True, exist_ok=True)
    errors: list[str] = []
    for url, digest in candidates:
        try:
            written = _download(url, destination, digest)
        except (OSError, urllib.error.URLError) as exc:
            errors.append(f"{url}: {exc}")
            continue
        print(f"wrote {written} ({_sha1(written)})")
        return 0
    print("download failed:", file=sys.stderr)
    for line in errors:
        print(f"  {line}", file=sys.stderr)
    return 1


def _candidates(urls: list[str] | None, sha1: str | None) -> list[tuple[str, str | None]]:
    if urls:
        return [(url, sha1.lower() if sha1 else None) for url in urls]
    return [(PRIMARY_URL, PRIMARY_SHA1), (MIRROR_URL, None)]


def _matches_any(path: Path, candidates: list[tuple[str, str | None]]) -> bool:
    digests = {digest for _url, digest in candidates if digest}
    if not digests:
        return True
    return _sha1(path) in digests


def _download(url: str, destination: Path, digest: str | None) -> Path:
    print(f"downloading {url}", file=sys.stderr)
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with tempfile.NamedTemporaryFile(delete=False, dir=destination.parent, suffix=".partial") as handle:
        partial = Path(handle.name)
        try:
            with urllib.request.urlopen(request, timeout=120) as response:
                while True:
                    chunk = response.read(256 * 1024)
                    if not chunk:
                        break
                    handle.write(chunk)
        except Exception:
            partial.unlink(missing_ok=True)
            raise
    actual = _sha1(partial)
    if digest is not None and actual != digest.lower():
        partial.unlink(missing_ok=True)
        raise OSError(f"sha1 mismatch for {url}: expected {digest}, got {actual}")
    partial.replace(destination)
    return destination


def _sha1(path: Path) -> str:
    digest = hashlib.sha1()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


if __name__ == "__main__":
    raise SystemExit(main())
