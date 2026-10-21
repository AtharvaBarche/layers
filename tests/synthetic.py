"""Controlled drawing with known pipes, crossings, and instruments."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from layer_sep.ocr_stage import parse_kind
from layer_sep.types import TextItem

WIDTH = 1400
HEIGHT = 900
THICK_Y = 250
THIN_Y = 560
THICK_W = 9
THIN_W = 3
THICK_CIRCLE = (520, THICK_Y, 42)
THIN_CIRCLE = (1080, THIN_Y, 36)
RECT = (190, 330, 80, 80)  # x, y, w, h
RECT_CX = 230
DIAMOND = (900, THICK_Y, 34)
CROSS_X = 760
THIN_BRANCH_X = 430
AMBIG_Y = 700
SCRAP = ((40, 90), (66, 90))
DISTANT_CIRCLE = (1240, 120, 26)
GAP_CIRCLE = (1180, 430, 28)
GAP_STUB_END = 340


@dataclass
class Scene:
    path: Path
    texts: list[TextItem]
    thick_gt: np.ndarray
    thin_gt: np.ndarray


def _text(index: int, text: str, bbox: tuple[float, float, float, float], angle: float = 0.0) -> TextItem:
    return TextItem(
        id=f"T-{index:04d}",
        text=text,
        bbox=bbox,
        confidence=0.95,
        kind=parse_kind(text),
        angle=angle,
        source="injected",
    )


def build_scene(folder: Path) -> Scene:
    folder.mkdir(parents=True, exist_ok=True)
    image = np.full((HEIGHT, WIDTH, 3), 255, np.uint8)
    thick_gt = np.zeros((HEIGHT, WIDTH), np.uint8)
    thin_gt = np.zeros((HEIGHT, WIDTH), np.uint8)

    def ink(target: np.ndarray, a, b, thickness: int) -> None:
        cv2.line(image, a, b, (0, 0, 0), thickness, lineType=cv2.LINE_8)
        cv2.line(target, a, b, 255, thickness, lineType=cv2.LINE_8)

    tcx, tcy, tr = THICK_CIRCLE
    ncx, ncy, nr = THIN_CIRCLE
    ink(thick_gt, (40, THICK_Y), (tcx - tr, THICK_Y), THICK_W)
    ink(thick_gt, (tcx + tr, THICK_Y), (1320, THICK_Y), THICK_W)
    ink(thick_gt, (RECT_CX, THICK_Y), (RECT_CX, RECT[1]), THICK_W)
    ink(thick_gt, (1180, THICK_Y), (1180, GAP_STUB_END), THICK_W)
    ink(thin_gt, (40, THIN_Y), (ncx - nr, THIN_Y), THIN_W)
    ink(thin_gt, (ncx + nr, THIN_Y), (1320, THIN_Y), THIN_W)
    ink(thin_gt, (RECT_CX, RECT[1] + RECT[3]), (RECT_CX, THIN_Y), THIN_W)
    ink(thin_gt, (THIN_BRANCH_X, THIN_Y), (THIN_BRANCH_X, 800), THIN_W)
    ink(thin_gt, (CROSS_X, 40), (CROSS_X, 860), THIN_W)

    cv2.line(image, SCRAP[0], SCRAP[1], (0, 0, 0), THICK_W, lineType=cv2.LINE_8)
    cv2.line(image, (200, AMBIG_Y), (1100, AMBIG_Y), (0, 0, 0), 6, lineType=cv2.LINE_8)

    cv2.circle(image, (tcx, tcy), tr, (0, 0, 0), 2, lineType=cv2.LINE_8)
    cv2.circle(image, (ncx, ncy), nr, (0, 0, 0), 2, lineType=cv2.LINE_8)
    cv2.circle(image, DISTANT_CIRCLE[:2], DISTANT_CIRCLE[2], (0, 0, 0), 2, lineType=cv2.LINE_8)
    cv2.circle(image, GAP_CIRCLE[:2], GAP_CIRCLE[2], (0, 0, 0), 2, lineType=cv2.LINE_8)
    x, y, w, h = RECT
    cv2.rectangle(image, (x, y), (x + w, y + h), (0, 0, 0), 2, lineType=cv2.LINE_8)
    cv2.line(image, (x + 18, y + h // 2), (x + w - 18, y + h // 2), (0, 0, 0), 2, lineType=cv2.LINE_8)
    cv2.line(image, (x + w // 2, y + 18), (x + w // 2, y + h - 18), (0, 0, 0), 2, lineType=cv2.LINE_8)

    dx, dy, dr = DIAMOND
    diamond = np.array([[dx, dy - dr], [dx + dr, dy], [dx, dy + dr], [dx - dr, dy]], np.int32)
    cv2.polylines(image, [diamond], True, (0, 0, 0), 2, lineType=cv2.LINE_8)
    cv2.circle(image, (dx + 38, dy), 22, (0, 0, 0), 2, lineType=cv2.LINE_8)

    cv2.putText(image, "DN150", (640, 225), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1, cv2.LINE_AA)
    cv2.putText(image, "6 bar", (110, 535), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1, cv2.LINE_AA)
    cv2.putText(image, "FT-101", (498, 254), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1, cv2.LINE_AA)
    cv2.putText(image, "SEE NOTE", (40, 860), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1, cv2.LINE_AA)

    rng = np.random.default_rng(0)
    for _ in range(30):
        image[int(rng.integers(0, HEIGHT)), int(rng.integers(0, WIDTH))] = 0

    path = folder / "scene.png"
    cv2.imwrite(str(path), image)
    texts = [
        _text(1, "DN150", (630, 190, 760, 258), 0.0),
        _text(2, "6 bar", (90, 500, 220, 548), 0.0),
        _text(3, "FT-101", (495, 235, 548, 268), 0.0),
        _text(4, "SEE NOTE", (30, 830, 190, 875), 0.0),
    ]
    return Scene(path=path, texts=texts, thick_gt=thick_gt, thin_gt=thin_gt)
