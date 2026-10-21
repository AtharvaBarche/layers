"""PaddleOCR text regions. Geometry still detects instruments without text."""

from __future__ import annotations

import os
import re
from typing import Any

import numpy as np

from layer_sep.config import LayerConfig
from layer_sep.geometry import undirected_angle
from layer_sep.types import TextItem

os.environ.setdefault("FLAGS_use_mkldnn", "0")
os.environ.setdefault("FLAGS_enable_pir_api", "0")
os.environ.setdefault("FLAGS_enable_pir_in_executor", "0")
os.environ.setdefault("PADDLE_PDX_ENABLE_MKLDNN_BYDEFAULT", "0")

_ENGINE = None

_PRESSURE = re.compile(r"(\d+(?:\.\d+)?)\s*(bar|psi|kpa|mpa|kg/cm2|kgcm2)\b", re.I)
_TEMP = re.compile(r"(-?\d+(?:\.\d+)?)\s*(degc|°c|c|°f|f)\b", re.I)
_FLOW = re.compile(r"(\d+(?:\.\d+)?)\s*(m3/h|m³/h|nm3/h|gpm|lpm|t/h|kg/h)\b", re.I)
_DIAMETER = re.compile(r"(?:DN|NPS|NB)\s*-?\s*\d+(?:\.\d+)?|\d+(?:\.\d+)?\s*(?:mm|in|inch|\"|''|NB|DN)\b", re.I)
_TAG = re.compile(r"^[A-Z]{1,4}\s*-?\s*\d{2,5}[A-Z]?$", re.I)


def parse_kind(text: str) -> str:
    raw = (text or "").strip()
    compact = re.sub(r"\s+", "", raw)
    if _PRESSURE.search(raw):
        return "PRESSURE"
    if _FLOW.search(raw):
        return "FLOW"
    if _TEMP.search(raw):
        return "TEMPERATURE"
    if re.search(r"DN|NPS|NB|mm|inch", raw, re.I) and _DIAMETER.search(raw):
        return "DIAMETER"
    if _TAG.match(compact):
        return "TAG"
    return "OTHER"


def _engine():
    global _ENGINE
    if _ENGINE is None:
        from paddleocr import PaddleOCR

        try:
            _ENGINE = PaddleOCR(
                lang="en",
                use_textline_orientation=True,
                use_doc_orientation_classify=False,
                use_doc_unwarping=False,
                enable_mkldnn=False,
                device="cpu",
            )
        except TypeError:
            _ENGINE = PaddleOCR(use_angle_cls=True, lang="en", show_log=False)
    return _ENGINE


def _rows_from_v3(res: Any) -> list[tuple[str, float, list]]:
    rec_texts = res.get("rec_texts") if isinstance(res, dict) else getattr(res, "rec_texts", None)
    rec_scores = res.get("rec_scores") if isinstance(res, dict) else getattr(res, "rec_scores", None)
    polys = res.get("dt_polys") if isinstance(res, dict) else getattr(res, "dt_polys", None)
    if rec_texts is None and hasattr(res, "json"):
        data = res.json
        if isinstance(data, dict):
            rec_texts = data.get("rec_texts")
            rec_scores = data.get("rec_scores")
            polys = data.get("dt_polys")
    if not rec_texts:
        return []
    rec_scores = rec_scores or [1.0] * len(rec_texts)
    polys = polys or []
    return [(str(text), float(score), poly) for text, score, poly in zip(rec_texts, rec_scores, polys)]


def _rows_from_v2(raw: Any) -> list[tuple[str, float, list]]:
    if not raw:
        return []
    page = raw[0] if isinstance(raw, list) and raw and isinstance(raw[0], list) else raw
    rows = []
    for item in page or []:
        if not item or len(item) < 2:
            continue
        box, rec = item[0], item[1]
        rows.append((str(rec[0]), float(rec[1]), box))
    return rows


def _points(poly: Any) -> list[tuple[float, float]]:
    pts = []
    for p in np.array(poly).reshape(-1, 2):
        pts.append((float(p[0]), float(p[1])))
    return pts


def _bbox(pts: list[tuple[float, float]]) -> tuple[float, float, float, float]:
    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    return (min(xs), min(ys), max(xs), max(ys))


def _angle(pts: list[tuple[float, float]]) -> float:
    if len(pts) < 2:
        return 0.0
    d01 = dist_pts(pts[0], pts[1])
    d03 = dist_pts(pts[0], pts[-1]) if len(pts) > 2 else 0.0
    if d01 >= d03:
        return undirected_angle(pts[0], pts[1])
    return undirected_angle(pts[0], pts[-1])


def dist_pts(a: tuple[float, float], b: tuple[float, float]) -> float:
    return ((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2) ** 0.5


def read_text(image: np.ndarray, cfg: LayerConfig) -> tuple[list[TextItem], str | None]:
    try:
        engine = _engine()
    except Exception as exc:
        return [], f"PaddleOCR unavailable: {exc}"
    rows: list[tuple[str, float, list]] = []
    try:
        if hasattr(engine, "predict"):
            out = engine.predict(image)
            first = out[0] if isinstance(out, list) and out else out
            rows = _rows_from_v3(first)
        if not rows and hasattr(engine, "ocr"):
            rows = _rows_from_v2(engine.ocr(image))
    except Exception as exc:
        return [], f"PaddleOCR failed: {exc}"
    items: list[TextItem] = []
    for text, score, poly in rows:
        text = (text or "").strip()
        if not text or score < cfg.ocr_min_conf:
            continue
        pts = _points(poly) if poly is not None and len(poly) else []
        if not pts:
            continue
        items.append(
            TextItem(
                id=f"T-{len(items) + 1:04d}",
                text=text,
                bbox=_bbox(pts),
                confidence=float(score),
                kind=parse_kind(text),
                angle=_angle(pts),
                source="paddleocr",
            )
        )
    return items, None
