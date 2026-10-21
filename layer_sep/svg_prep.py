"""Prepare the thick-pipe layer for a later SVG conversion.

Cleanup is written beside the raw mask. The raw mask is left intact.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from layer_sep.config import LayerConfig
from layer_sep.geometry import rdp
from layer_sep.types import Branch


def _drop_small(mask: np.ndarray, min_area: int) -> np.ndarray:
    count, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    cleaned = np.zeros_like(mask)
    for i in range(1, count):
        if int(stats[i, cv2.CC_STAT_AREA]) >= min_area:
            cleaned[labels == i] = 255
    return cleaned


def prepare_thick_svg(
    thick_mask: np.ndarray,
    branches: list[Branch],
    cfg: LayerConfig,
    out_dir: Path,
) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    raw_path = out_dir / "thick_pipe_mask_raw.png"
    cleaned_path = out_dir / "thick_pipe_mask_cleaned.png"
    cv2.imwrite(str(raw_path), thick_mask)
    cleaned = _drop_small(thick_mask, cfg.cleanup_min_area)
    cv2.imwrite(str(cleaned_path), cleaned)

    height, width = thick_mask.shape[:2]
    centerlines = []
    for branch in branches:
        if branch.label != "thick_pipe" or len(branch.pixels) < 2:
            continue
        points = [(float(x), float(y)) for x, y in branch.pixels]
        simplified = rdp(points, cfg.rdp_epsilon)
        centerlines.append(
            {
                "id": branch.id,
                "confidence": round(branch.confidence, 4),
                "width": round(branch.width, 3),
                "points": [[round(x, 2), round(y, 2)] for x, y in simplified],
            }
        )
    contours, _ = cv2.findContours(cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    contour_rows = []
    for index, contour in enumerate(contours, start=1):
        approx = cv2.approxPolyDP(contour, cfg.contour_epsilon, closed=True)
        pts = [[int(p[0][0]), int(p[0][1])] for p in approx]
        if len(pts) < 3:
            continue
        contour_rows.append({"id": f"TC-{index:04d}", "points": pts})
    manifest = {
        "image_width": width,
        "image_height": height,
        "coordinate_system": "original_image_pixels",
        "raw_mask": raw_path.name,
        "cleaned_mask": cleaned_path.name,
        "centerline_count": len(centerlines),
        "contour_count": len(contour_rows),
        "smoothing": {
            "centerline_rdp_epsilon_px": cfg.rdp_epsilon,
            "contour_epsilon_px": cfg.contour_epsilon,
            "note": "Epsilon stays at 1 px so junctions are not moved.",
        },
        "instrument_mask": "../instrument_mask.png",
        "connections": "../connections.json",
    }
    _write(out_dir / "thick_centerlines.json", {"image_width": width, "image_height": height, "centerlines": centerlines})
    _write(out_dir / "thick_contours.json", {"image_width": width, "image_height": height, "contours": contour_rows})
    _write(out_dir / "manifest.json", manifest)
    return manifest


def _write(path: Path, payload: dict) -> None:
    import json

    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
