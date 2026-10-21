"""Associate OCR text with pipes. Overlap with a text box does not delete a pipe."""

from __future__ import annotations

import numpy as np

from layer_sep.config import LayerConfig, ResolvedThresholds
from layer_sep.geometry import angle_diff_deg, clamp, dist
from layer_sep.types import Branch, Instrument, TextItem


def _unclamped_t(
    px: float, py: float, x1: float, y1: float, x2: float, y2: float
) -> tuple[float, float]:
    vx, vy = x2 - x1, y2 - y1
    length2 = vx * vx + vy * vy
    if length2 <= 1e-9:
        return dist((px, py), (x1, y1)), 0.0
    t = ((px - x1) * vx + (py - y1) * vy) / length2
    t_clamped = max(0.0, min(1.0, t))
    qx = x1 + t_clamped * vx
    qy = y1 + t_clamped * vy
    return dist((px, py), (qx, qy)), t


def _inside(bbox: tuple[float, float, float, float], point: tuple[float, float], pad: float = 2.0) -> bool:
    x1, y1, x2, y2 = bbox
    x, y = point
    return (x1 - pad) <= x <= (x2 + pad) and (y1 - pad) <= y <= (y2 + pad)


def link_tags(texts: list[TextItem], instruments: list[Instrument]) -> None:
    accepted = [item for item in instruments if item.status == "accepted"]
    for text in texts:
        if text.kind != "TAG":
            continue
        for instrument in accepted:
            if _inside(text.bbox, instrument.center, pad=0) or _center_in_instrument(text, instrument):
                instrument.ocr_ids.append(text.id)
                text.instrument_id = instrument.id
                text.association_status = "instrument_tag"
                text.association_confidence = float(text.confidence)
                text.association_reason = "tag_inside_instrument"
                break


def _center_in_instrument(text: TextItem, instrument: Instrument) -> bool:
    x, y, w, h = instrument.bbox
    cx, cy = text.center
    return x - 2 <= cx <= x + w + 2 and y - 2 <= cy <= y + h + 2


def associate_text(
    texts: list[TextItem],
    branches: list[Branch],
    cfg: LayerConfig,
    resolved: ResolvedThresholds,
) -> None:
    pipes = [branch for branch in branches if branch.label in {"thick_pipe", "thin_pipe"}]
    for text in texts:
        if text.association_status == "instrument_tag":
            continue
        scored: list[tuple[float, Branch, dict]] = []
        for branch in pipes:
            score, evidence = _score_pipe(text, branch, resolved.text_max_dist)
            if score <= 0:
                continue
            scored.append((score, branch, evidence))
        scored.sort(key=lambda item: item[0], reverse=True)
        if not scored:
            text.association_status = "unresolved"
            text.association_reason = "no_pipe_in_range"
            text.association_confidence = 0.0
            continue
        best, branch, evidence = scored[0]
        second = scored[1][0] if len(scored) > 1 else 0.0
        margin = best - second
        evidence["second_best"] = round(second, 4)
        evidence["margin"] = round(margin, 4)
        if best >= cfg.text_assign_min_conf and margin >= cfg.text_assign_margin:
            text.pipe_id = branch.id
            text.association_status = "assigned"
            text.association_confidence = float(best)
            text.association_reason = "geometry_and_kind"
        else:
            text.association_status = "unresolved"
            text.association_confidence = float(best)
            text.association_reason = "ambiguous_pipe_candidates" if margin < cfg.text_assign_margin else "low_confidence"
        text.association_reason = text.association_reason


def _score_pipe(text: TextItem, branch: Branch, max_dist: float) -> tuple[float, dict]:
    x1, y1 = branch.start
    x2, y2 = branch.end
    cx, cy = text.center
    distance, t = _unclamped_t(cx, cy, x1, y1, x2, y2)
    if distance > max_dist:
        return 0.0, {}
    along = 1.0 if -0.05 <= t <= 1.05 else clamp(1.0 - abs(t - 0.5) / 2.0)
    orientation = clamp(1.0 - angle_diff_deg(text.angle, branch.orientation) / 45.0)
    distance_score = clamp(1.0 - distance / max_dist)
    kind_score = 1.0 if text.kind in {"DIAMETER", "PRESSURE", "TEMPERATURE", "FLOW"} else 0.4
    score = 0.36 * distance_score + 0.26 * orientation + 0.22 * along + 0.16 * kind_score
    return score, {
        "distance": round(distance, 3),
        "projection_t": round(t, 3),
        "orientation_score": round(orientation, 3),
        "kind": text.kind,
    }


def raster_text(
    shape: tuple[int, int],
    geometric: np.ndarray,
    texts: list[TextItem],
) -> np.ndarray:
    import cv2

    mask = geometric.copy() if geometric is not None else np.zeros(shape, np.uint8)
    for text in texts:
        x1, y1, x2, y2 = [int(round(v)) for v in text.bbox]
        cv2.rectangle(mask, (x1, y1), (x2, y2), 255, thickness=-1)
    return mask
