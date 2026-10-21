"""Geometric instrument candidates. OCR is not required."""

from __future__ import annotations

import math

import cv2
import numpy as np

from layer_sep.config import LayerConfig, ResolvedThresholds
from layer_sep.geometry import clamp
from layer_sep.types import Instrument


def _shape_scores(contour: np.ndarray, circularity_min: float) -> dict[str, float]:
    area = float(cv2.contourArea(contour))
    peri = float(cv2.arcLength(contour, True))
    if peri < 1.0 or area < 1.0:
        return {}
    circularity = 4.0 * math.pi * area / (peri * peri)
    approx = cv2.approxPolyDP(contour, 0.04 * peri, True)
    x, y, w, h = cv2.boundingRect(contour)
    aspect = w / max(h, 1)
    fill = area / max(float(w * h), 1.0)
    scores: dict[str, float] = {}
    if circularity >= circularity_min and 0.72 <= aspect <= 1.4:
        scores["circle"] = clamp(circularity)
    if len(contour) >= 5 and circularity >= 0.55 and 0.35 <= aspect <= 2.8:
        scores["ellipse"] = clamp(circularity * 0.92)
    if len(approx) == 4 and 0.45 <= aspect <= 2.2:
        pts = approx.reshape(-1, 2).astype(np.float32)
        corners = np.array([[x, y], [x + w, y], [x + w, y + h], [x, y + h]], np.float32)
        mids = np.array(
            [[x + w / 2, y], [x + w, y + h / 2], [x + w / 2, y + h], [x, y + h / 2]],
            np.float32,
        )
        corner_err = _mean_nearest(pts, corners) / max(w, h)
        mid_err = _mean_nearest(pts, mids) / max(w, h)
        if corner_err < 0.2 and fill > 0.62:
            scores["rectangle"] = clamp(1.0 - corner_err)
        if mid_err < 0.2 and fill < 0.78:
            scores["diamond"] = clamp(1.0 - mid_err)
    if 5 <= len(approx) <= 10 and circularity > 0.4:
        scores["polygon"] = clamp(0.5 + 0.3 * circularity)
    return {name: score for name, score in scores.items() if score >= 0.5}


def _mean_nearest(points: np.ndarray, targets: np.ndarray) -> float:
    total = 0.0
    for point in points:
        delta = targets - point
        total += float(np.min(np.hypot(delta[:, 0], delta[:, 1])))
    return total / max(len(points), 1)


def _internal_ink(binary: np.ndarray, contour: np.ndarray) -> float:
    mask = np.zeros(binary.shape[:2], np.uint8)
    cv2.drawContours(mask, [contour], -1, 255, thickness=-1)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    interior = cv2.erode(mask, kernel)
    interior_count = int(np.count_nonzero(interior))
    if interior_count < 8:
        return 0.0
    ink = int(np.count_nonzero(cv2.bitwise_and(binary, interior)))
    return ink / float(interior_count)


def _mark_grids(candidates: list[Instrument], min_count: int) -> None:
    rects = [item for item in candidates if item.shape == "rectangle"]
    if len(rects) < min_count:
        return
    used = [False] * len(rects)
    for i, item in enumerate(rects):
        if used[i]:
            continue
        x, y, w, h = item.bbox
        group = [i]
        for j in range(i + 1, len(rects)):
            if used[j]:
                continue
            ox, oy, ow, oh = rects[j].bbox
            aligned = abs(ox - x) <= 4 or abs(oy - y) <= 4 or abs((ox + ow) - (x + w)) <= 4
            similar = abs(ow - w) <= max(8, 0.25 * w) and abs(oh - h) <= max(8, 0.25 * h)
            if aligned and similar:
                group.append(j)
        if len(group) >= min_count:
            for index in group:
                used[index] = True
                rects[index].in_grid = True
                rects[index].reasons.append("aligned_empty_grid")
                rects[index].confidence = clamp(rects[index].confidence - 0.28)


def _ring_fraction(binary: np.ndarray, x: float, y: float, radius: float) -> float:
    height, width = binary.shape[:2]
    hits = 0
    samples = 48
    for i in range(samples):
        ang = 2.0 * math.pi * i / samples
        px = int(round(x + radius * math.cos(ang)))
        py = int(round(y + radius * math.sin(ang)))
        y0, y1 = max(0, py - 1), min(height, py + 2)
        x0, x1 = max(0, px - 1), min(width, px + 2)
        if y1 > y0 and x1 > x0 and int(binary[y0:y1, x0:x1].max()) > 0:
            hits += 1
    return hits / float(samples)


def _circle_contour(x: float, y: float, radius: float) -> np.ndarray:
    pts = cv2.ellipse2Poly(
        (int(round(x)), int(round(y))),
        (int(round(radius)), int(round(radius))),
        0,
        0,
        360,
        8,
    )
    return pts.reshape(-1, 1, 2)


def _hough_instruments(binary: np.ndarray, resolved: ResolvedThresholds, cfg: LayerConfig) -> list[Instrument]:
    min_r = max(15, int(round(resolved.min_symbol_side * 0.7)))
    max_r = max(min_r + 4, int(round(resolved.max_symbol_side * 0.55)))
    circles = cv2.HoughCircles(
        binary,
        cv2.HOUGH_GRADIENT,
        dp=1.2,
        minDist=max(24, min_r),
        param1=80,
        param2=22,
        minRadius=min_r,
        maxRadius=max_r,
    )
    found: list[Instrument] = []
    if circles is None:
        return found
    for x, y, radius in circles[0]:
        ring = _ring_fraction(binary, float(x), float(y), float(radius))
        if ring < 0.72:
            continue
        contour = _circle_contour(float(x), float(y), float(radius))
        side = int(round(radius * 2))
        bbox = (int(round(x - radius)), int(round(y - radius)), side, side)
        ink = _internal_ink(binary, contour)
        confidence = clamp(0.55 + 0.45 * ring)
        reasons = ["hough_circle", f"ring:{ring:.2f}"]
        if ink >= 0.02:
            reasons.append("internal_ink")
        found.append(
            Instrument(
                id="",
                shape="circle",
                bbox=bbox,
                center=(float(x), float(y)),
                confidence=float(confidence),
                contour=contour,
                status="accepted" if confidence >= cfg.instrument_min_confidence else "weak",
                internal_ink=float(ink),
                reasons=reasons,
            )
        )
    return found


def _face_instruments(skeleton: np.ndarray, binary: np.ndarray, resolved: ResolvedThresholds, cfg: LayerConfig) -> list[Instrument]:
    ys, xs = np.nonzero(skeleton)
    if len(xs) == 0:
        return []
    pixels = set(zip(xs.tolist(), ys.tolist()))
    neighbors: dict[tuple[int, int], list[tuple[int, int]]] = {}
    for x, y in pixels:
        found = []
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                if dx == 0 and dy == 0:
                    continue
                q = (x + dx, y + dy)
                if q in pixels:
                    found.append(q)
        neighbors[(x, y)] = found

    def _next(prev: tuple[int, int], cur: tuple[int, int]) -> tuple[int, int]:
        options = [q for q in neighbors[cur] if q != prev] or [prev]
        bx, by = prev[0] - cur[0], prev[1] - cur[1]
        best = options[0]
        best_ang = -99.0
        for cand in options:
            wx, wy = cand[0] - cur[0], cand[1] - cur[1]
            ang = math.atan2(bx * wy - by * wx, bx * wx + by * wy)
            if ang > best_ang:
                best_ang = ang
                best = cand
        return best

    used: set[tuple[tuple[int, int], tuple[int, int]]] = set()
    found: list[Instrument] = []
    limit = 1500
    for start, nbs in neighbors.items():
        for nxt in nbs:
            if (start, nxt) in used:
                continue
            face = [start]
            prev, cur = start, nxt
            guard = 0
            closed = False
            while guard < limit:
                used.add((prev, cur))
                face.append(cur)
                nxt_pt = _next(prev, cur)
                prev, cur = cur, nxt_pt
                guard += 1
                if (prev, cur) == (start, nxt):
                    closed = True
                    break
            if not closed or len(face) < 40:
                continue
            arr = np.array(face, np.int32)
            x0, y0 = int(arr[:, 0].min()), int(arr[:, 1].min())
            x1, y1 = int(arr[:, 0].max()), int(arr[:, 1].max())
            w, h = x1 - x0 + 1, y1 - y0 + 1
            side = max(w, h)
            if side < max(28, resolved.min_symbol_side) or side > resolved.max_symbol_side:
                continue
            if w / max(h, 1) > 2.4 or h / max(w, 1) > 2.4:
                continue
            contour = arr.reshape(-1, 1, 2)
            scores = _shape_scores(contour, cfg.circle_circularity)
            if not scores:
                continue
            shape = max(scores, key=scores.get)
            if shape not in {"rectangle", "diamond", "circle", "polygon"}:
                continue
            ink = _internal_ink(binary, contour)
            confidence = scores[shape]
            reasons = [f"skeleton_face:{shape}"]
            if ink >= 0.02:
                confidence = clamp(confidence + 0.08)
                reasons.append("internal_ink")
            elif shape == "rectangle":
                confidence = clamp(confidence - 0.2)
                reasons.append("empty_rectangle")
            if confidence < cfg.instrument_record_confidence:
                continue
            status = "accepted" if confidence >= cfg.instrument_min_confidence and not (shape == "rectangle" and ink < 0.02) else "weak"
            found.append(
                Instrument(
                    id="",
                    shape=shape,
                    bbox=(x0, y0, w, h),
                    center=((x0 + x1) / 2.0, (y0 + y1) / 2.0),
                    confidence=float(confidence),
                    contour=contour,
                    status=status,
                    internal_ink=float(ink),
                    reasons=reasons,
                )
            )
    return found


def _reject_header_rows(items: list[Instrument], height: int) -> None:
    """A row of similar circles in the title strip is lettering, not instruments."""
    circles = [item for item in items if item.shape == "circle"]
    band = max(14.0, 0.02 * height)
    top = 0.16 * height
    for item in circles:
        if item.center[1] > top:
            continue
        mates = [
            other
            for other in circles
            if other.center[1] <= top and abs(other.center[1] - item.center[1]) <= band
        ]
        if len(mates) < 3:
            continue
        for mate in mates:
            mate.status = "rejected"
            if "header_row" not in mate.reasons:
                mate.reasons.append("header_row")


def _dedupe(items: list[Instrument]) -> list[Instrument]:
    kept: list[Instrument] = []
    for item in sorted(items, key=lambda inst: inst.confidence, reverse=True):
        side = max(item.bbox[2], item.bbox[3])
        if any(math.hypot(item.center[0] - other.center[0], item.center[1] - other.center[1]) < 0.55 * max(side, other.bbox[2], other.bbox[3]) for other in kept):
            continue
        kept.append(item)
    return kept


def detect_instruments(
    binary: np.ndarray,
    skeleton: np.ndarray,
    cfg: LayerConfig,
    resolved: ResolvedThresholds,
) -> list[Instrument]:
    contours, hierarchy = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    found: list[Instrument] = []
    hierarchy = None if hierarchy is None else hierarchy[0]
    serial = 0
    height, width = binary.shape[:2]
    for index, contour in enumerate(contours):
        if hierarchy is None or hierarchy[index][3] != -1:
            continue
        x, y, w, h = cv2.boundingRect(contour)
        side = max(w, h)
        # Isolated contours miss symbols that touch a pipe. Small letter-like
        # blobs are left to the text mask. Hough and skeleton faces cover the rest.
        if side < max(resolved.min_symbol_side, 32) or side > resolved.max_symbol_side:
            continue
        if w < 8 or h < 8:
            continue
        if x <= 1 or y <= 1 or x + w >= width - 1 or y + h >= height - 1:
            continue
        aspect = w / max(h, 1)
        if aspect > 3.2 or aspect < 0.32:
            continue
        scores = _shape_scores(contour, cfg.circle_circularity)
        if not scores:
            continue
        shape = max(scores, key=scores.get)
        children = _child_count(hierarchy, index)
        ink = _internal_ink(binary, contour)
        confidence = scores[shape]
        reasons = [f"shape:{shape}"]
        if shape == "circle":
            confidence = clamp(confidence + 0.08)
            reasons.append("closed_circle")
        elif shape == "diamond":
            confidence = clamp(confidence + 0.06)
            reasons.append("diamond_vertices")
        elif shape == "ellipse" and ink < 0.01 and children == 0:
            confidence = clamp(confidence - 0.08)
        if children > 0:
            confidence = clamp(confidence + 0.05)
            reasons.append("internal_contour")
        if ink >= 0.02:
            confidence = clamp(confidence + 0.08)
            reasons.append("internal_ink")
        elif shape == "rectangle":
            confidence = clamp(confidence - 0.22)
            reasons.append("empty_rectangle")
        if confidence < cfg.instrument_record_confidence:
            continue
        serial += 1
        status = "accepted" if confidence >= cfg.instrument_min_confidence else "weak"
        if shape == "rectangle" and ink < 0.02:
            status = "weak"
        found.append(
            Instrument(
                id=f"I-{serial:04d}",
                shape=shape,
                bbox=(int(x), int(y), int(w), int(h)),
                center=(x + w / 2.0, y + h / 2.0),
                confidence=float(confidence),
                contour=contour,
                status=status,
                internal_ink=float(ink),
                child_count=children,
                reasons=reasons,
            )
        )
    found.extend(_hough_instruments(binary, resolved, cfg))
    found.extend(_face_instruments(skeleton, binary, resolved, cfg))
    found = _dedupe(found)
    _reject_header_rows(found, binary.shape[0])
    _mark_grids(found, cfg.grid_min_count)
    for item in found:
        if item.in_grid:
            item.status = "weak"
            item.confidence = clamp(item.confidence)
    for serial, item in enumerate(found, start=1):
        item.id = f"I-{serial:04d}"
    return found


def _child_count(hierarchy: np.ndarray, index: int) -> int:
    count = 0
    child = int(hierarchy[index][2])
    while child != -1 and count < 12:
        count += 1
        child = int(hierarchy[child][0])
    return count


def raster_instruments(shape: tuple[int, int], instruments: list[Instrument]) -> np.ndarray:
    mask = np.zeros(shape, np.uint8)
    for item in instruments:
        if item.status != "accepted":
            continue
        cv2.drawContours(mask, [item.contour], -1, 255, thickness=-1)
    if int(mask.max()) == 0:
        return mask
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    return cv2.dilate(mask, kernel, iterations=1)
