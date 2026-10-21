"""Debug pictures for each major stage."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from layer_sep.types import Connection, Instrument, TextItem


def _write(path: Path, image: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), image)


def save_gray(path: Path, gray: np.ndarray) -> None:
    _write(path, gray)


def save_mask(path: Path, mask: np.ndarray) -> None:
    _write(path, mask)


def save_width(path: Path, width_map: np.ndarray, binary: np.ndarray) -> None:
    positive = width_map[binary > 0]
    peak = float(np.percentile(positive, 95)) if positive.size else 1.0
    peak = max(peak, 1.0)
    scaled = np.clip(width_map / peak * 255.0, 0, 255).astype(np.uint8)
    color = cv2.applyColorMap(scaled, cv2.COLORMAP_VIRIDIS)
    color[binary == 0] = 0
    _write(path, color)


def save_classes(path: Path, shape: tuple[int, int], layers: dict[str, np.ndarray]) -> None:
    colors = {
        "thick": (255, 80, 0),
        "thin": (0, 140, 255),
        "ambiguous": (180, 180, 180),
        "text": (0, 220, 255),
        "instrument": (0, 180, 0),
        "frame": (80, 80, 80),
    }
    canvas = np.zeros((shape[0], shape[1], 3), np.uint8)
    for name, mask in layers.items():
        if mask is None:
            continue
        canvas[mask > 0] = colors.get(name, (255, 255, 255))
    _write(path, canvas)


def save_instruments(path: Path, original: np.ndarray, instruments: list[Instrument]) -> None:
    canvas = original.copy()
    for item in instruments:
        color = (0, 180, 0) if item.status == "accepted" else (0, 165, 255)
        cv2.drawContours(canvas, [item.contour], -1, color, 2)
        x, y = int(item.center[0]), int(item.center[1])
        cv2.putText(canvas, item.id, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1, cv2.LINE_AA)
    _write(path, canvas)


def save_overlay(
    path: Path,
    original: np.ndarray,
    thick: np.ndarray,
    thin: np.ndarray,
    instruments: np.ndarray,
    text: np.ndarray,
    junctions: dict,
    connections: list[Connection],
    ambiguous: np.ndarray | None = None,
) -> np.ndarray:
    overlay = original.copy()
    _tint(overlay, ambiguous, (180, 180, 180), 0.55)
    _tint(overlay, text, (0, 220, 255), 0.45)
    _tint(overlay, thin, (0, 140, 255), 0.7)
    _tint(overlay, thick, (255, 90, 0), 0.7)
    _tint(overlay, instruments, (0, 190, 0), 0.35)
    for node in junctions.get("junctions", []):
        _dot(overlay, node["xy"], (0, 0, 255), 4)
    for node in junctions.get("visual_crossings", []):
        _dot(overlay, node["xy"], (255, 0, 180), 4)
    for node in junctions.get("unresolved_crossings", []):
        _dot(overlay, node["xy"], (0, 200, 255), 4)
    for node in junctions.get("endpoints", []):
        color = (255, 0, 255) if node.get("role") == "instrument_port" else (0, 255, 0)
        _dot(overlay, node["xy"], color, 3)
    for connection in connections:
        if connection.status != "verified":
            continue
        x, y = int(round(connection.port_xy[0])), int(round(connection.port_xy[1]))
        cv2.circle(overlay, (x, y), 6, (255, 255, 255), 1, cv2.LINE_AA)
    _write(path, overlay)
    return overlay


def save_comparison(path: Path, original: np.ndarray, overlay: np.ndarray) -> None:
    if original.shape[:2] != overlay.shape[:2]:
        overlay = cv2.resize(overlay, (original.shape[1], original.shape[0]), interpolation=cv2.INTER_NEAREST)
    gap = np.full((original.shape[0], 8, 3), 255, np.uint8)
    side = np.hstack([original, gap, overlay])
    _write(path, side)


def _tint(image: np.ndarray, mask: np.ndarray, color: tuple[int, int, int], alpha: float) -> None:
    if mask is None or int(mask.max()) == 0:
        return
    sel = mask > 0
    src = image[sel].astype(np.float32)
    paint = np.array(color, dtype=np.float32)
    image[sel] = np.clip(src * (1.0 - alpha) + paint * alpha, 0, 255).astype(np.uint8)


def _dot(image: np.ndarray, xy: list | tuple, color: tuple[int, int, int], radius: int) -> None:
    x, y = int(round(xy[0])), int(round(xy[1]))
    cv2.circle(image, (x, y), radius, color, -1, cv2.LINE_AA)


def draw_text_boxes(path: Path, original: np.ndarray, texts: list[TextItem]) -> None:
    canvas = original.copy()
    for item in texts:
        x1, y1, x2, y2 = [int(round(v)) for v in item.bbox]
        color = (0, 200, 0) if item.association_status == "assigned" else (0, 180, 255)
        cv2.rectangle(canvas, (x1, y1), (x2, y2), color, 1)
    _write(path, canvas)
