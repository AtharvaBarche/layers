"""CLI for the layer-separation module.

Example:
    python run_layers.py path\\to\\diagram.png --skip-ocr
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

os.environ.setdefault("FLAGS_use_mkldnn", "0")
os.environ.setdefault("FLAGS_enable_pir_api", "0")
os.environ.setdefault("FLAGS_enable_pir_in_executor", "0")
os.environ.setdefault("PADDLE_PDX_ENABLE_MKLDNN_BYDEFAULT", "0")

from layer_sep.config import LayerConfig
from layer_sep.pipeline import separate_layers

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".webp"}


def collect_images(src: Path) -> list[Path]:
    if src.is_file():
        if src.suffix.lower() not in IMAGE_EXTS:
            raise SystemExit(f"Not an image: {src}")
        return [src]
    if not src.is_dir():
        raise SystemExit(f"Path not found: {src}")
    images = sorted(
        path
        for path in src.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_EXTS and "output" not in path.parts
    )
    if not images:
        raise SystemExit(f"No images in {src}")
    return images


def main() -> None:
    parser = argparse.ArgumentParser(description="Separate a P&ID into thick pipes, thin pipes, instruments, and text.")
    parser.add_argument("path", type=Path, help="Image file or folder")
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("output") / "layer_separation",
        help="Dedicated output directory. Existing pipeline folders are not overwritten.",
    )
    parser.add_argument("--skip-ocr", action="store_true", help="Skip PaddleOCR")
    parser.add_argument("--no-deskew", action="store_true", help="Do not rotate the working copy")
    args = parser.parse_args()
    cfg = LayerConfig(deskew=not args.no_deskew)
    for image in collect_images(args.path.resolve()):
        result = separate_layers(image, args.out, cfg=cfg, skip_ocr=args.skip_ocr)
        counts = result.validation["counts"]
        print(
            f"Wrote {result.output_dir} | "
            f"thick branches {counts['branches'].get('thick_pipe', 0)} | "
            f"thin branches {counts['branches'].get('thin_pipe', 0)} | "
            f"instruments {counts['instruments_accepted']} | "
            f"verified {counts['verified_connections']} | "
            f"unresolved {counts['unverified_connections']}"
        )


if __name__ == "__main__":
    main()
