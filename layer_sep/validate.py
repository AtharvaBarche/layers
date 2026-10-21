"""Validation report. Accuracy stays null unless a ground-truth mask is supplied."""

from __future__ import annotations

import numpy as np

from layer_sep.evaluate import pixel_scores
from layer_sep.types import Branch, Connection, Instrument, TextItem


def _mean(values: list[float]) -> float | None:
    if not values:
        return None
    return round(sum(values) / len(values), 4)


def _ink_near(binary: np.ndarray, xy: tuple[float, float], radius: int = 3) -> bool:
    x = int(round(xy[0]))
    y = int(round(xy[1]))
    height, width = binary.shape[:2]
    y0, y1 = max(0, y - radius), min(height, y + radius + 1)
    x0, x1 = max(0, x - radius), min(width, x + radius + 1)
    patch = binary[y0:y1, x0:x1]
    return bool(patch.size) and int(patch.max()) > 0


def build_report(
    *,
    image_path: str,
    width: int,
    height: int,
    original_sha256: str,
    original_intact: bool,
    branches: list[Branch],
    instruments: list[Instrument],
    connections: list[Connection],
    texts: list[TextItem],
    junctions: dict,
    binary: np.ndarray,
    thick_mask: np.ndarray,
    thin_mask: np.ndarray,
    warnings: list[str],
    ocr_warning: str | None,
    skew_deg: float,
    thin_peak: float | None,
    thick_peak: float | None,
    ground_truth: dict[str, np.ndarray] | None = None,
) -> dict:
    labels: dict[str, int] = {}
    for branch in branches:
        labels[branch.label] = labels.get(branch.label, 0) + 1
    verified = [item for item in connections if item.status == "verified"]
    unverified = [item for item in connections if item.status == "unverified"]
    rejected = [item for item in connections if item.status == "rejected"]
    report_warnings = list(warnings)
    if ocr_warning:
        report_warnings.append(ocr_warning)
    if labels.get("thick_pipe", 0) == 0:
        report_warnings.append("No branch passed the thick-pipe gates.")
    if labels.get("thin_pipe", 0) == 0:
        report_warnings.append("No branch passed the thin-pipe gates.")
    for item in verified:
        if not _ink_near(binary, item.port_xy):
            report_warnings.append(
                f"{item.id} port has no ink within 3 px of the original binary."
            )
    for group_name in ("junctions", "visual_crossings", "unresolved_crossings", "endpoints"):
        for node in junctions.get(group_name, []):
            xy = node.get("xy")
            if xy and not _ink_near(binary, (xy[0], xy[1])):
                report_warnings.append(f"{node.get('id')} is not on ink in the original binary.")
    pipe_conf = [branch.confidence for branch in branches if branch.label in {"thick_pipe", "thin_pipe"}]
    inst_conf = [item.confidence for item in instruments if item.status == "accepted"]
    accuracy = {
        "pixel_iou": None,
        "connection_f1": None,
        "available": False,
        "note": (
            "No manually verified ground truth was provided. "
            "Pixel accuracy and association accuracy were not measured, and high accuracy is not claimed."
        ),
    }
    per_mask = None
    if ground_truth:
        per_mask = {}
        for name, truth in ground_truth.items():
            predicted = {"thick_pipe_mask": thick_mask, "thin_pipe_mask": thin_mask}.get(name)
            if predicted is None:
                continue
            per_mask[name] = pixel_scores(predicted, truth)
        accuracy = {
            "pixel_iou": per_mask,
            "connection_f1": None,
            "available": True,
            "note": "Scores are only for the ground-truth masks that were passed in.",
        }
    accepted = [item for item in instruments if item.status == "accepted"]
    return {
        "image": {
            "path": image_path,
            "width": width,
            "height": height,
            "sha256": original_sha256,
            "original_unchanged": original_intact,
            "skew_deg_applied": round(skew_deg, 3),
        },
        "width_modes": {
            "thin_peak": None if thin_peak is None else round(thin_peak, 3),
            "thick_peak": None if thick_peak is None else round(thick_peak, 3),
        },
        "counts": {
            "branches": labels,
            "instruments_accepted": len(accepted),
            "instruments_recorded": len(instruments),
            "text_regions": len(texts),
            "endpoints": len(junctions.get("endpoints", [])),
            "junctions": len(junctions.get("junctions", [])),
            "visual_crossings": len(junctions.get("visual_crossings", [])),
            "unresolved_crossings": len(junctions.get("unresolved_crossings", [])),
            "verified_connections": len(verified),
            "unverified_connections": len(unverified),
            "rejected_nearby": len(rejected),
            "thick_pixels": int(np.count_nonzero(thick_mask)),
            "thin_pixels": int(np.count_nonzero(thin_mask)),
        },
        "confidence": {
            "pipe_mean": _mean(pipe_conf),
            "instrument_mean": _mean(inst_conf),
            "verified_connection_mean": _mean([item.confidence for item in verified]),
            "text_association_mean": _mean(
                [item.association_confidence for item in texts if item.association_status == "assigned"]
            ),
        },
        "unresolved_connections": [item.summary() for item in unverified],
        "endpoints": junctions.get("endpoints", []),
        "junctions": junctions.get("junctions", []),
        "visual_crossings": junctions.get("visual_crossings", []),
        "unresolved_crossings": junctions.get("unresolved_crossings", []),
        "warnings": report_warnings,
        "accuracy": accuracy,
    }
