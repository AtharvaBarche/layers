"""Separate pixel-mask scores from connectivity scores.

These numbers are measurements. They are not a claim about a drawing that
has no manually verified ground truth.
"""

from __future__ import annotations

import math

import numpy as np


def pixel_scores(predicted: np.ndarray, truth: np.ndarray) -> dict:
    pred = predicted > 0
    gt = truth > 0
    if pred.shape != gt.shape:
        raise ValueError(f"mask shapes differ: {pred.shape} vs {gt.shape}")
    tp = int(np.count_nonzero(pred & gt))
    fp = int(np.count_nonzero(pred & ~gt))
    fn = int(np.count_nonzero(~pred & gt))
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    iou = tp / (tp + fp + fn) if (tp + fp + fn) else 0.0
    return {
        "precision": precision,
        "recall": recall,
        "iou": iou,
        "tp": tp,
        "fp": fp,
        "fn": fn,
    }


def connection_scores(
    predicted: list[tuple[float, float, str]],
    truth: list[tuple[float, float, str]],
    tolerance_px: float = 20.0,
) -> dict:
    """Match (x, y, class) links. A nearby wrong class is not a true positive."""
    used: set[int] = set()
    tp = 0
    for px, py, plabel in predicted:
        for index, (tx, ty, tlabel) in enumerate(truth):
            if index in used or plabel != tlabel:
                continue
            if math.hypot(px - tx, py - ty) <= tolerance_px:
                used.add(index)
                tp += 1
                break
    fp = len(predicted) - tp
    fn = len(truth) - len(used)
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    return {
        "precision": precision,
        "recall": recall,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tolerance_px": tolerance_px,
    }
