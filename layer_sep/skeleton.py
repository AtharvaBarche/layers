"""Skeleton tracing. Branches stop at endpoints, junctions, and corners."""

from __future__ import annotations

import cv2
import numpy as np
from skimage.morphology import skeletonize

from layer_sep.geometry import dist, polyline_length, undirected_angle


def skeleton_image(mask: np.ndarray) -> np.ndarray:
    if mask is None or mask.size == 0 or int(mask.max()) == 0:
        return np.zeros(mask.shape[:2], np.uint8)
    skel = skeletonize(mask > 0)
    return (skel.astype(np.uint8) * 255)


def _neighbors(point: tuple[int, int], pixels: set[tuple[int, int]]) -> list[tuple[int, int]]:
    x, y = point
    found = []
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            if dx == 0 and dy == 0:
                continue
            candidate = (x + dx, y + dy)
            if candidate in pixels:
                found.append(candidate)
    found.sort()
    return found


def _components(pixels: set[tuple[int, int]]) -> list[set[tuple[int, int]]]:
    seen: set[tuple[int, int]] = set()
    groups: list[set[tuple[int, int]]] = []
    for start in sorted(pixels):
        if start in seen:
            continue
        stack = [start]
        seen.add(start)
        group: set[tuple[int, int]] = set()
        while stack:
            current = stack.pop()
            group.add(current)
            for nxt in _neighbors(current, pixels):
                if nxt not in seen:
                    seen.add(nxt)
                    stack.append(nxt)
        groups.append(group)
    return groups


def _edge(a: tuple[int, int], b: tuple[int, int]) -> tuple[tuple[int, int], tuple[int, int]]:
    return (a, b) if a <= b else (b, a)


def _trace_from(
    start: tuple[int, int],
    first: tuple[int, int],
    pixels: set[tuple[int, int]],
    special: set[tuple[int, int]],
    visited: set[tuple[tuple[int, int], tuple[int, int]]],
) -> list[tuple[int, int]]:
    path = [start, first]
    visited.add(_edge(start, first))
    prev, cur = start, first
    guard = 0
    limit = len(pixels) + 5
    while cur not in special:
        nxts = [
            q
            for q in _neighbors(cur, pixels)
            if q != prev and _edge(cur, q) not in visited
        ]
        if not nxts:
            break
        nxt = nxts[0]
        visited.add(_edge(cur, nxt))
        prev, cur = cur, nxt
        path.append(cur)
        guard += 1
        if guard > limit:
            break
    return path


def trace_component(pixels: set[tuple[int, int]]) -> list[list[tuple[int, int]]]:
    if len(pixels) < 2:
        return []
    degrees = {p: len(_neighbors(p, pixels)) for p in pixels}
    special = {p for p, degree in degrees.items() if degree != 2}
    if not special:
        special.add(min(pixels))
    visited: set[tuple[tuple[int, int], tuple[int, int]]] = set()
    branches: list[list[tuple[int, int]]] = []
    for start in sorted(special):
        for nxt in _neighbors(start, pixels):
            if _edge(start, nxt) in visited:
                continue
            path = _trace_from(start, nxt, pixels, special, visited)
            if len(path) >= 2:
                branches.append(path)
    return branches


def split_at_corners(path: list[tuple[int, int]], min_deg: float) -> list[list[tuple[int, int]]]:
    if len(path) < 8:
        return [path]
    step = 4
    cuts = [0]
    last = 0
    i = step
    while i < len(path) - step:
        ang = _turn(path[i - step], path[i], path[i + step])
        if ang >= min_deg and i - last >= step:
            cuts.append(i)
            last = i
            i += step
        else:
            i += 1
    cuts.append(len(path) - 1)
    parts: list[list[tuple[int, int]]] = []
    for a, b in zip(cuts, cuts[1:]):
        if b > a:
            parts.append(path[a : b + 1])
    return parts or [path]


def _turn(a: tuple[int, int], b: tuple[int, int], c: tuple[int, int]) -> float:
    """Degrees of turn along the path. A straight run is 0."""
    import math

    v1 = (b[0] - a[0], b[1] - a[1])
    v2 = (c[0] - b[0], c[1] - b[1])
    n1 = (v1[0] ** 2 + v1[1] ** 2) ** 0.5
    n2 = (v2[0] ** 2 + v2[1] ** 2) ** 0.5
    if n1 < 1e-6 or n2 < 1e-6:
        return 0.0
    cos = (v1[0] * v2[0] + v1[1] * v2[1]) / (n1 * n2)
    cos = max(-1.0, min(1.0, cos))
    return math.degrees(math.acos(cos))


def trace_mask(mask: np.ndarray, corner_deg: float) -> list[tuple[str, list[tuple[int, int]]]]:
    skel = skeleton_image(mask)
    ys, xs = np.nonzero(skel)
    pixels = set(zip(xs.tolist(), ys.tolist()))
    traced: list[tuple[str, list[tuple[int, int]]]] = []
    for index, component in enumerate(_components(pixels), start=1):
        cid = f"CC-{index:04d}"
        for path in trace_component(component):
            for part in split_at_corners(path, corner_deg):
                if len(part) >= 2:
                    traced.append((cid, part))
    return traced


def path_length(pixels: list[tuple[int, int]]) -> float:
    return polyline_length([(float(x), float(y)) for x, y in pixels])


def linearity(pixels: list[tuple[int, int]]) -> float:
    if len(pixels) < 3:
        return 1.0
    ax, ay = pixels[0]
    bx, by = pixels[-1]
    span = dist((ax, ay), (bx, by))
    if span < 1.0:
        return 0.0
    from layer_sep.geometry import point_to_segment

    worst = 0.0
    for x, y in pixels:
        deviation, _ = point_to_segment(x, y, ax, ay, bx, by)
        if deviation > worst:
            worst = deviation
    return max(0.0, 1.0 - worst / (0.12 * span + 1.0))


def orientation_of(pixels: list[tuple[int, int]]) -> float:
    if len(pixels) < 2:
        return 0.0
    return undirected_angle(pixels[0], pixels[-1])


def width_along(pixels: list[tuple[int, int]], width_map: np.ndarray) -> tuple[float, float]:
    values = []
    height, width = width_map.shape[:2]
    for x, y in pixels:
        if 0 <= x < width and 0 <= y < height:
            value = float(width_map[y, x])
            if value > 0.2:
                values.append(value)
    if not values:
        return 0.0, 0.0
    arr = np.array(values, dtype=np.float32)
    return float(np.median(arr)), float(np.std(arr))


def split_by_mask(
    path: list[tuple[int, int]],
    inside: np.ndarray,
) -> list[tuple[bool, list[tuple[int, int]]]]:
    """Cut a path where it enters or leaves a mask. The boundary pixel is shared."""
    if len(path) < 2:
        return []
    height, width = inside.shape[:2]
    flags = []
    for x, y in path:
        flags.append(bool(0 <= x < width and 0 <= y < height and inside[y, x] > 0))
    parts: list[tuple[bool, list[tuple[int, int]]]] = []
    start = 0
    for i in range(1, len(path)):
        if flags[i] != flags[i - 1]:
            parts.append((flags[start], path[start : i + 1]))
            start = i
    parts.append((flags[start], path[start:]))
    return [(flag, pts) for flag, pts in parts if len(pts) >= 2]
