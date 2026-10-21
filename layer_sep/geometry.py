"""Geometric helpers shared by the layer modules."""

from __future__ import annotations

import math


def clamp(value: float, lo: float = 0.0, hi: float = 1.0) -> float:
    if value < lo:
        return lo
    if value > hi:
        return hi
    return value


def dist(a: tuple[float, float], b: tuple[float, float]) -> float:
    return math.hypot(float(a[0]) - float(b[0]), float(a[1]) - float(b[1]))


def undirected_angle(a: tuple[float, float], b: tuple[float, float]) -> float:
    ang = math.degrees(math.atan2(float(b[1]) - float(a[1]), float(b[0]) - float(a[0])))
    return ang % 180.0


def directed_angle(a: tuple[float, float], b: tuple[float, float]) -> float:
    ang = math.degrees(math.atan2(float(b[1]) - float(a[1]), float(b[0]) - float(a[0])))
    return ang % 360.0


def angle_diff_deg(a: float, b: float) -> float:
    """Smallest difference between undirected orientations, in degrees."""
    delta = abs(a - b) % 180.0
    return min(delta, 180.0 - delta)


def directed_diff(a: float, b: float) -> float:
    delta = abs(a - b) % 360.0
    return min(delta, 360.0 - delta)


def point_to_segment(
    px: float,
    py: float,
    x1: float,
    y1: float,
    x2: float,
    y2: float,
) -> tuple[float, float]:
    """Return distance and clamped projection parameter t in [0, 1]."""
    vx, vy = x2 - x1, y2 - y1
    length2 = vx * vx + vy * vy
    if length2 <= 1e-9:
        return dist((px, py), (x1, y1)), 0.0
    t = ((px - x1) * vx + (py - y1) * vy) / length2
    t_clamped = max(0.0, min(1.0, t))
    qx = x1 + t_clamped * vx
    qy = y1 + t_clamped * vy
    return dist((px, py), (qx, qy)), t_clamped


def polyline_length(points: list[tuple[float, float]]) -> float:
    if len(points) < 2:
        return 0.0
    total = 0.0
    for a, b in zip(points, points[1:]):
        total += dist(a, b)
    return total


def rdp(points: list[tuple[float, float]], epsilon: float) -> list[tuple[float, float]]:
    """Ramer-Douglas-Peucker. Epsilon is in pixels and should stay small."""
    if len(points) < 3 or epsilon <= 0:
        return list(points)
    x1, y1 = points[0]
    x2, y2 = points[-1]
    max_d = -1.0
    index = 0
    for i in range(1, len(points) - 1):
        d, _ = point_to_segment(points[i][0], points[i][1], x1, y1, x2, y2)
        if d > max_d:
            max_d = d
            index = i
    if max_d > epsilon:
        left = rdp(points[: index + 1], epsilon)
        right = rdp(points[index:], epsilon)
        return left[:-1] + right
    return [points[0], points[-1]]


def axis_deviation(angle_deg: float) -> float:
    """Signed degrees from the nearest axis. Range is [-45, 45]."""
    folded = angle_deg % 90.0
    if folded > 45.0:
        folded -= 90.0
    return folded
