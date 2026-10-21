"""Full-resolution preprocessing. The source array is not painted."""

from __future__ import annotations

import math
from dataclasses import dataclass

import cv2
import numpy as np

from layer_sep.config import LayerConfig
from layer_sep.geometry import axis_deviation


@dataclass
class PreprocessResult:
    gray: np.ndarray
    binary: np.ndarray
    width_map: np.ndarray
    skew_deg: float
    to_original: np.ndarray | None
    working_shape: tuple[int, int]


def _ink_binary(gray: np.ndarray, cfg: LayerConfig) -> np.ndarray:
    work = gray
    if float(np.mean(gray)) > 127.0:
        work = cv2.bitwise_not(gray)
    ksize = int(cfg.denoise_ksize)
    if ksize >= 3:
        if ksize % 2 == 0:
            ksize += 1
        blur = cv2.GaussianBlur(work, (ksize, ksize), 0)
    else:
        blur = work
    _, otsu = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    block = max(3, int(cfg.adaptive_block))
    if block % 2 == 0:
        block += 1
    adaptive = cv2.adaptiveThreshold(
        blur,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        block,
        int(cfg.adaptive_c),
    )
    agreed = cv2.bitwise_and(otsu, adaptive)
    binary = agreed if float(np.mean(agreed)) >= 0.45 * float(np.mean(otsu)) else otsu
    count, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    cleaned = np.zeros_like(binary)
    min_area = max(1, int(cfg.min_component_area))
    for i in range(1, count):
        if int(stats[i, cv2.CC_STAT_AREA]) >= min_area:
            cleaned[labels == i] = 255
    return cleaned


def estimate_skew_deg(binary: np.ndarray, cfg: LayerConfig) -> float:
    """Median axis error of long near-axis lines. Diagonals are ignored."""
    lines = cv2.HoughLinesP(
        binary,
        rho=1,
        theta=np.pi / 180.0,
        threshold=max(20, int(cfg.hough_threshold)),
        minLineLength=max(20, int(cfg.hough_min_length)),
        maxLineGap=int(cfg.hough_max_gap),
    )
    if lines is None:
        return 0.0
    deviations: list[float] = []
    for x1, y1, x2, y2 in lines[:, 0]:
        ang = math.degrees(math.atan2(float(y2 - y1), float(x2 - x1))) % 180.0
        dev = axis_deviation(ang)
        if abs(dev) <= 8.0:
            deviations.append(dev)
    if len(deviations) < cfg.deskew_min_lines:
        return 0.0
    return float(np.median(np.array(deviations, dtype=np.float32)))


def _rotate(image: np.ndarray, angle_deg: float, border: int) -> tuple[np.ndarray, np.ndarray]:
    height, width = image.shape[:2]
    center = (width * 0.5, height * 0.5)
    matrix = cv2.getRotationMatrix2D(center, angle_deg, 1.0)
    rotated = cv2.warpAffine(
        image,
        matrix,
        (width, height),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=border,
    )
    return rotated, cv2.invertAffineTransform(matrix)


def preprocess(original_bgr: np.ndarray, cfg: LayerConfig) -> PreprocessResult:
    if original_bgr.ndim == 2:
        gray = original_bgr.copy()
    else:
        gray = cv2.cvtColor(original_bgr, cv2.COLOR_BGR2GRAY)
    probe = _ink_binary(gray, cfg)
    skew = estimate_skew_deg(probe, cfg) if cfg.deskew else 0.0
    to_original = None
    working = gray
    if cfg.deskew and cfg.deskew_min_deg <= abs(skew) <= cfg.deskew_max_deg:
        # Positive OpenCV angle is counter-clockwise. Subtract the skew.
        working, to_original = _rotate(gray, -skew, border=255)
    binary = _ink_binary(working, cfg)
    distance = cv2.distanceTransform(binary, cv2.DIST_L2, 5)
    width_map = distance * 2.0
    return PreprocessResult(
        gray=working,
        binary=binary,
        width_map=width_map,
        skew_deg=float(skew if to_original is not None else 0.0),
        to_original=to_original,
        working_shape=working.shape[:2],
    )


def map_mask_to_original(mask: np.ndarray, prepared: PreprocessResult) -> np.ndarray:
    if prepared.to_original is None:
        return mask
    height, width = prepared.working_shape
    return cv2.warpAffine(
        mask,
        prepared.to_original,
        (width, height),
        flags=cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )


def map_point_to_original(point: tuple[float, float], prepared: PreprocessResult) -> tuple[float, float]:
    if prepared.to_original is None:
        return (float(point[0]), float(point[1]))
    x, y = float(point[0]), float(point[1])
    m = prepared.to_original
    ox = m[0, 0] * x + m[0, 1] * y + m[0, 2]
    oy = m[1, 0] * x + m[1, 1] * y + m[1, 2]
    return (ox, oy)
