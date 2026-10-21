"""Classify skeleton branches with width, continuity, and line geometry.

Width selects a candidate class. Length, linearity, and directional support
decide whether that candidate is actually a pipe. The middle of a bimodal
width distribution stays ambiguous.
"""

from __future__ import annotations

import math

import cv2
import numpy as np

from layer_sep.config import LayerConfig, ResolvedThresholds, resolve_thresholds
from layer_sep.geometry import angle_diff_deg, clamp, point_to_segment
from layer_sep.geometry import dist
from layer_sep.skeleton import linearity, orientation_of, path_length, split_by_mask, width_along
from layer_sep.types import Branch


def find_width_peaks(widths: list[float], min_count: int) -> tuple[float | None, float | None]:
    if len(widths) < min_count:
        return None, None
    arr = np.array(widths, dtype=np.float32)
    hi = max(float(arr.max()), 2.0)
    bins = np.linspace(0.4, hi + 0.6, max(8, int(hi * 2)))
    hist, edges = np.histogram(arr, bins=bins)
    kernel = np.array([1, 2, 3, 2, 1], dtype=np.float32)
    kernel /= kernel.sum()
    smooth = np.convolve(hist.astype(np.float32), kernel, mode="same")
    peaks: list[tuple[float, float, int]] = []
    for i in range(1, len(smooth) - 1):
        if smooth[i] >= smooth[i - 1] and smooth[i] > smooth[i + 1] and hist[i] >= min_count:
            center = 0.5 * (edges[i] + edges[i + 1])
            peaks.append((float(center), float(smooth[i]), int(hist[i])))
    if not peaks:
        return float(np.median(arr)), None
    peaks.sort(key=lambda item: item[1], reverse=True)
    primary = peaks[0][0]
    secondary = None
    for center, _score, _count in peaks[1:]:
        base = max(min(center, primary), 0.5)
        if abs(center - primary) / base >= 0.40:
            secondary = center
            break
    if secondary is None:
        return primary, None
    thin, thick = sorted((primary, secondary))
    return thin, thick


def width_class(
    width: float,
    thin: float | None,
    thick: float | None,
    cfg: LayerConfig,
) -> tuple[str, float]:
    """Return thick, thin, thinner, or ambiguous, plus a width confidence."""
    if thin is None or width <= 0:
        return "ambiguous", 0.0
    if thick is None:
        tol = max(cfg.width_abs_tol, cfg.width_rel_tol * thin)
        if abs(width - thin) <= tol:
            name = "thick" if thin >= cfg.single_mode_thick_min else "thin"
            return name, clamp(1.0 - abs(width - thin) / max(tol, 0.8))
        return "ambiguous", 0.25
    band = max(cfg.ambiguous_band_px, cfg.ambiguous_band_ratio * (thick - thin))
    if (thin + band) < width < (thick - band):
        return "ambiguous", 0.2
    if abs(width - thin) <= abs(width - thick):
        if width <= thin + band:
            return "thin", clamp(1.0 - abs(width - thin) / max(band, 0.8))
        return "ambiguous", 0.2
    if width >= thick - band:
        return "thick", clamp(1.0 - abs(width - thick) / max(band, 0.8))
    return "ambiguous", 0.2


def _line_kernel(length: int, angle_deg: float) -> np.ndarray:
    length = max(3, int(length))
    if length % 2 == 0:
        length += 1
    kernel = np.zeros((length, length), np.uint8)
    center = length // 2
    theta = math.radians(angle_deg)
    for step in range(-center, center + 1):
        x = int(round(center + step * math.cos(theta)))
        y = int(round(center + step * math.sin(theta)))
        if 0 <= x < length and 0 <= y < length:
            kernel[y, x] = 1
    return kernel


def directional_mask(binary: np.ndarray, kernel_length: int) -> np.ndarray:
    combined = np.zeros_like(binary)
    for angle in (0.0, 45.0, 90.0, 135.0):
        kernel = _line_kernel(kernel_length, angle)
        opened = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
        combined = cv2.bitwise_or(combined, opened)
    return combined


def hough_segments(mask: np.ndarray, cfg: LayerConfig) -> list[tuple[float, float, float, float]]:
    lines = cv2.HoughLinesP(
        mask,
        rho=1,
        theta=np.pi / 180.0,
        threshold=int(cfg.hough_threshold),
        minLineLength=max(8, int(cfg.hough_min_length)),
        maxLineGap=int(cfg.hough_max_gap),
    )
    if lines is None:
        return []
    return [tuple(float(v) for v in row) for row in lines[:, 0]]


def _hough_support(
    branch: Branch,
    segments: list[tuple[float, float, float, float]],
    cfg: LayerConfig,
) -> bool:
    if len(branch.pixels) < 2 or not segments:
        return False
    mid = branch.pixels[len(branch.pixels) // 2]
    for x1, y1, x2, y2 in segments:
        orientation = math.degrees(math.atan2(y2 - y1, x2 - x1)) % 180.0
        if angle_diff_deg(branch.orientation, orientation) > cfg.hough_angle_tol:
            continue
        distance, _ = point_to_segment(mid[0], mid[1], x1, y1, x2, y2)
        if distance <= cfg.hough_dist_px:
            return True
    return False


def _measure(branch: Branch, width_map: np.ndarray) -> None:
    branch.length = path_length(branch.pixels)
    branch.linearity = linearity(branch.pixels)
    branch.orientation = orientation_of(branch.pixels)
    branch.width, branch.width_std = width_along(branch.pixels, width_map)


def _text_component_mask(
    binary: np.ndarray,
    instrument_mask: np.ndarray,
    resolved: ResolvedThresholds,
) -> np.ndarray:
    count, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    mask = np.zeros(binary.shape[:2], np.uint8)
    max_h = resolved.text_max_height
    for i in range(1, count):
        x, y, w, h, area = [int(v) for v in stats[i]]
        if h > max_h or w > max_h * 4 or area < 4:
            continue
        if max(w, h) > resolved.min_pipe_length * 0.65:
            continue
        aspect = w / max(h, 1)
        if aspect > 8.0:
            continue
        component = labels == i
        if instrument_mask is not None and int(np.count_nonzero(component & (instrument_mask > 0))) > 0.6 * area:
            continue
        mask[component] = 255
    return mask


def _near_frame(branch: Branch, shape: tuple[int, int], cfg: LayerConfig) -> bool:
    height, width = shape
    margin = cfg.frame_margin_px
    xs = [p[0] for p in (branch.start, branch.end)]
    ys = [p[1] for p in (branch.start, branch.end)]
    horizontal = abs(branch.start[1] - branch.end[1]) <= margin and (
        min(ys) <= margin or max(ys) >= height - 1 - margin
    )
    vertical = abs(branch.start[0] - branch.end[0]) <= margin and (
        min(xs) <= margin or max(xs) >= width - 1 - margin
    )
    span = branch.length
    if horizontal and span >= cfg.frame_span_fraction * width:
        return True
    if vertical and span >= cfg.frame_span_fraction * height:
        return True
    return False


def _overlap_fraction(pixels: list[tuple[int, int]], mask: np.ndarray) -> float:
    if not pixels or mask is None:
        return 0.0
    height, width = mask.shape[:2]
    hits = 0
    for x, y in pixels:
        if 0 <= x < width and 0 <= y < height and mask[y, x] > 0:
            hits += 1
    return hits / float(len(pixels))


def bridge_collinear_pipes(
    branches: list[Branch],
    width_map: np.ndarray,
    cluster_px: float,
    angle_tol: float,
) -> list[Branch]:
    """Join two same-class pipes that meet with no third pipe.

    A four-way crossing is left split so it can be judged as a crossing.
    """
    others = [branch for branch in branches if branch.label not in {"thick_pipe", "thin_pipe"}]
    pipes = [branch for branch in branches if branch.label in {"thick_pipe", "thin_pipe"}]
    changed = True
    guard = 0
    while changed and guard < 10:
        changed = False
        guard += 1
        items: list[tuple[tuple[int, int], Branch]] = []
        for branch in pipes:
            items.append((branch.start, branch))
            if branch.end != branch.start:
                items.append((branch.end, branch))
        parent = list(range(len(items)))

        def find(i: int) -> int:
            while parent[i] != i:
                parent[i] = parent[parent[i]]
                i = parent[i]
            return i

        def union(i: int, j: int) -> None:
            ri, rj = find(i), find(j)
            if ri != rj:
                parent[rj] = ri

        for i in range(len(items)):
            for j in range(i + 1, len(items)):
                if dist(items[i][0], items[j][0]) <= cluster_px:
                    union(i, j)
        groups: dict[int, list[tuple[tuple[int, int], Branch]]] = {}
        for index, item in enumerate(items):
            groups.setdefault(find(index), []).append(item)
        merged_ids: set[str] = set()
        additions: list[Branch] = []
        for group in groups.values():
            unique: list[Branch] = []
            seen: set[str] = set()
            for _pt, branch in group:
                if branch.id not in seen:
                    seen.add(branch.id)
                    unique.append(branch)
            if len(unique) != 2:
                continue
            left, right = unique
            if left.label != right.label:
                continue
            if angle_diff_deg(left.orientation, right.orientation) > angle_tol:
                continue
            if left.id in merged_ids or right.id in merged_ids:
                continue
            pixels = _concatenate(left, right, cluster_px)
            if pixels is None:
                continue
            merged_ids.add(left.id)
            merged_ids.add(right.id)
            joined = Branch(
                id=left.id,
                pixels=pixels,
                component_id=left.component_id,
                label=left.label,
                confidence=max(left.confidence, right.confidence),
                reasons=left.reasons + ["bridged_collinear"],
                inside_instrument=None,
            )
            _measure(joined, width_map)
            additions.append(joined)
            changed = True
        if changed:
            pipes = [branch for branch in pipes if branch.id not in merged_ids] + additions
    return others + pipes


def _concatenate(left: Branch, right: Branch, cluster_px: float) -> list[tuple[int, int]] | None:
    options = [
        (left.pixels, right.pixels, dist(left.end, right.start)),
        (left.pixels, list(reversed(right.pixels)), dist(left.end, right.end)),
        (list(reversed(left.pixels)), right.pixels, dist(left.start, right.start)),
        (list(reversed(left.pixels)), list(reversed(right.pixels)), dist(left.start, right.end)),
    ]
    options.sort(key=lambda item: item[2])
    first, second, gap = options[0]
    if gap > cluster_px + 1.0:
        return None
    return list(first) + list(second[1:])


def build_branches(
    raw_paths: list[tuple[str, list[tuple[int, int]]]],
    instrument_mask: np.ndarray,
    width_map: np.ndarray,
    id_start: int = 1,
) -> list[Branch]:
    branches: list[Branch] = []
    serial = id_start
    owners = _mask_ids(instrument_mask)
    for cid, path in raw_paths:
        parts = split_by_mask(path, instrument_mask) if int(instrument_mask.max()) > 0 else [(False, path)]
        if not parts:
            parts = [(False, path)]
        for inside, pixels in parts:
            branch = Branch(id=f"BR-{serial:04d}", pixels=pixels, component_id=cid)
            serial += 1
            _measure(branch, width_map)
            if inside:
                branch.inside_instrument = _owner_at(pixels, owners)
                branch.label = "instrument_stroke"
                branch.reasons.append("inside_instrument_mask")
            branches.append(branch)
    return branches


def _mask_ids(mask: np.ndarray) -> np.ndarray | None:
    return None


def _owner_at(pixels: list[tuple[int, int]], owners: np.ndarray | None) -> str:
    return "instrument"


def label_branches(
    branches: list[Branch],
    binary: np.ndarray,
    width_map: np.ndarray,
    cfg: LayerConfig,
    resolved: ResolvedThresholds,
    segments: list[tuple[float, float, float, float]],
    text_mask: np.ndarray,
    directional: np.ndarray,
) -> None:
    del width_map
    for branch in branches:
        if branch.label == "instrument_stroke":
            branch.confidence = 0.7
            continue
        if _near_frame(branch, binary.shape[:2], cfg):
            branch.label = "frame"
            branch.confidence = 0.4
            branch.reasons.append("page_frame")
            continue
        kind, width_score = width_class(branch.width, resolved.thin_peak, resolved.thick_peak, cfg)
        continuity = clamp(1.0 - branch.width_std / max(branch.width, 1.0))
        length_score = 0.0
        if branch.length >= resolved.min_pipe_length:
            length_score = clamp(branch.length / (resolved.min_pipe_length * 4.0))
        supported = _overlap_fraction(branch.pixels, directional) >= 0.35
        hough_ok = _hough_support(branch, segments, cfg)
        score = (
            cfg.w_width * width_score
            + cfg.w_length * length_score
            + cfg.w_linearity * branch.linearity
            + cfg.w_continuity * continuity
            + cfg.w_directional * (1.0 if supported or hough_ok else 0.0)
        )
        text_frac = _overlap_fraction(branch.pixels, text_mask)
        long_enough = branch.length >= resolved.min_pipe_length
        straight = branch.linearity >= cfg.min_linearity
        pipe_kind = kind in {"thick", "thin"}
        if (
            pipe_kind
            and long_enough
            and straight
            and score >= cfg.pipe_score_threshold
            and text_frac < 0.55
        ):
            branch.label = "thick_pipe" if kind == "thick" else "thin_pipe"
            branch.confidence = float(score)
            branch.reasons.append(f"width:{kind}")
            if supported:
                branch.reasons.append("directional_opening")
            if hough_ok:
                branch.reasons.append("hough_support")
            continue
        if branch.length <= resolved.spur_max:
            branch.label = "noise"
            branch.confidence = 0.2
            branch.reasons.append("short_spur")
            continue
        if text_frac >= 0.45 or (
            kind in {"thin", "thinner", "ambiguous"}
            and branch.length < resolved.min_pipe_length
            and text_frac > 0
        ):
            branch.label = "text"
            branch.confidence = max(0.45, text_frac)
            branch.reasons.append("text_component")
            continue
        branch.label = "ambiguous"
        branch.confidence = float(score)
        if not pipe_kind:
            branch.reasons.append(f"width:{kind}")
        if not long_enough:
            branch.reasons.append("below_min_length")
        if not straight:
            branch.reasons.append("not_linear")
        if text_frac >= 0.55 and pipe_kind:
            branch.reasons.append("text_overlap_blocked_pipe")
        if score < cfg.pipe_score_threshold:
            branch.reasons.append("score_below_threshold")


def classify_paths(
    raw_paths: list[tuple[str, list[tuple[int, int]]]],
    binary: np.ndarray,
    width_map: np.ndarray,
    instrument_mask: np.ndarray,
    cfg: LayerConfig,
    image_shape: tuple[int, int],
) -> tuple[list[Branch], ResolvedThresholds, np.ndarray, list[tuple[float, float, float, float]]]:
    probe = build_branches(raw_paths, instrument_mask, width_map)
    sample = [
        branch.width
        for branch in probe
        if branch.label != "instrument_stroke"
        and branch.linearity >= 0.68
        and branch.length >= cfg.min_pipe_length_px * 0.45
        and branch.width > 0.4
    ]
    thin, thick = find_width_peaks(sample, cfg.min_peak_count)
    if thin and thick and (thick - thin) < max(1.4, 0.45 * thin):
        thin, thick = (thick if thick > thin else thin), None
    resolved = resolve_thresholds(cfg, thin, thick, image_shape)
    # Peak estimation uses a provisional length. Re-estimate once the scaled
    # minimum length is known so tiny scraps do not define the modes.
    sample = [
        branch.width
        for branch in probe
        if branch.label != "instrument_stroke"
        and branch.linearity >= 0.68
        and branch.length >= resolved.min_pipe_length * 0.5
        and branch.width > 0.4
    ]
    thin, thick = find_width_peaks(sample, cfg.min_peak_count)
    if thin and thick and (thick - thin) < max(1.4, 0.45 * thin):
        thin, thick = max(thin, thick), None
    resolved = resolve_thresholds(cfg, thin, thick, image_shape)
    text_mask = _text_component_mask(binary, instrument_mask, resolved)
    directional = directional_mask(binary, resolved.direction_kernel)
    segments = hough_segments(binary, cfg)
    label_branches(probe, binary, width_map, cfg, resolved, segments, text_mask, directional)
    probe = bridge_collinear_pipes(probe, width_map, cfg.node_cluster_px, cfg.hough_angle_tol)
    return probe, resolved, text_mask, segments


def reconstruct_mask(
    shape: tuple[int, int],
    branches: list[Branch],
    labels: set[str],
    binary: np.ndarray,
    exclude: np.ndarray | None = None,
) -> np.ndarray:
    mask = np.zeros(shape, np.uint8)
    for branch in branches:
        if branch.label not in labels or len(branch.pixels) < 2:
            continue
        thickness = max(1, int(round(branch.width if branch.width > 0 else 1)))
        pts = np.array(branch.pixels, np.int32).reshape(-1, 1, 2)
        cv2.polylines(mask, [pts], False, 255, thickness=thickness, lineType=cv2.LINE_8)
    mask = cv2.bitwise_and(mask, binary)
    if exclude is not None:
        mask[exclude > 0] = 0
    return mask
