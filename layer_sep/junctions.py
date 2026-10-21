"""Skeleton topology: endpoints, branches, and visual crossings."""

from __future__ import annotations

import numpy as np

from layer_sep.config import LayerConfig
from layer_sep.geometry import angle_diff_deg, directed_angle, directed_diff, dist, point_to_segment
from layer_sep.types import Branch, Connection


def is_visual_crossing(angles: list[float], tol: float) -> bool:
    """True when four outgoing directions form two straight lines."""
    if len(angles) != 4:
        return False
    pairings = (((0, 1), (2, 3)), ((0, 2), (1, 3)), ((0, 3), (1, 2)))
    for (a, b), (c, d) in pairings:
        if _opposite(angles[a], angles[b], tol) and _opposite(angles[c], angles[d], tol):
            return True
    return False


def _opposite(a: float, b: float, tol: float) -> bool:
    return abs(directed_diff(a, b) - 180.0) <= tol


def _has_through_line(angles: list[float], tol: float) -> bool:
    for i in range(len(angles)):
        for j in range(i + 1, len(angles)):
            if _opposite(angles[i], angles[j], tol):
                return True
    return False


def _outgoing(branch: Branch, endpoint: tuple[int, int]) -> float:
    pts = branch.pixels
    if dist(pts[0], endpoint) > dist(pts[-1], endpoint):
        pts = list(reversed(pts))
    step = min(8, len(pts) - 1)
    if step < 1:
        return 0.0
    return directed_angle(pts[0], pts[step])


def _cluster(branches: list[Branch], radius: float) -> list[list[tuple[tuple[int, int], Branch]]]:
    items: list[tuple[tuple[int, int], Branch]] = []
    for branch in branches:
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
            if dist(items[i][0], items[j][0]) <= radius:
                union(i, j)
    groups: dict[int, list[tuple[tuple[int, int], Branch]]] = {}
    for index, item in enumerate(items):
        groups.setdefault(find(index), []).append(item)
    return list(groups.values())


def _local_width(width_map: np.ndarray, center: tuple[float, float]) -> float:
    x = int(round(center[0]))
    y = int(round(center[1]))
    height, width = width_map.shape[:2]
    y0, y1 = max(0, y - 2), min(height, y + 3)
    x0, x1 = max(0, x - 2), min(width, x + 3)
    patch = width_map[y0:y1, x0:x1]
    if patch.size == 0:
        return 0.0
    positive = patch[patch > 0.2]
    if positive.size == 0:
        return 0.0
    return float(np.median(positive))


def detect_junctions(
    branches: list[Branch],
    width_map: np.ndarray,
    cfg: LayerConfig,
) -> dict:
    pipes = [branch for branch in branches if branch.label in {"thick_pipe", "thin_pipe"}]
    endpoints: list[dict] = []
    junctions: list[dict] = []
    crossings: list[dict] = []
    unresolved: list[dict] = []
    serial = {"E": 0, "J": 0, "X": 0, "U": 0}

    def take(prefix: str) -> str:
        serial[prefix] += 1
        return f"{prefix}-{serial[prefix]:04d}"

    for group in _cluster(pipes, cfg.node_cluster_px):
        unique: list[Branch] = []
        seen: set[str] = set()
        points = []
        for point, branch in group:
            points.append(point)
            if branch.id not in seen:
                seen.add(branch.id)
                unique.append(branch)
        if not unique:
            continue
        cx = sum(p[0] for p in points) / len(points)
        cy = sum(p[1] for p in points) / len(points)
        center = (cx, cy)
        angles = []
        for point, branch in group:
            if branch.id in {b.id for b in unique}:
                angles.append(_outgoing(branch, point))
        # One angle per branch, from the endpoint that fell in this cluster.
        per_branch: dict[str, float] = {}
        for point, branch in group:
            per_branch[branch.id] = _outgoing(branch, point)
        angles = [per_branch[branch.id] for branch in unique]
        degree = len(unique)
        widths = [branch.width for branch in unique]
        reference = max(widths) if widths else 1.0
        local = _local_width(width_map, center)
        blob = local > cfg.junction_blob_ratio * max(reference, 0.5)
        record = {
            "xy": [round(cx, 2), round(cy, 2)],
            "degree": degree,
            "branch_ids": [branch.id for branch in unique],
            "width_ratio": round(local / max(reference, 0.5), 3),
            "angles": [round(a, 2) for a in angles],
        }
        if degree <= 1:
            record["id"] = take("E")
            record["kind"] = "endpoint"
            record["role"] = "free"
            record["confidence"] = 0.8
            endpoints.append(record)
        elif degree == 2:
            continue
        elif degree == 3:
            record["id"] = take("J")
            record["kind"] = "junction"
            record["confidence"] = 0.84 if _has_through_line(angles, cfg.crossing_angle_tol) else 0.62
            record["note"] = "branch" if _has_through_line(angles, cfg.crossing_angle_tol) else "three_way"
            junctions.append(record)
        elif degree == 4 and is_visual_crossing(angles, cfg.crossing_angle_tol) and not blob:
            record["id"] = take("X")
            record["kind"] = "visual_crossing"
            record["confidence"] = 0.74
            record["note"] = "collinear_pairs_without_junction_blob"
            crossings.append(record)
        else:
            record["id"] = take("U")
            record["kind"] = "unresolved_crossing"
            record["confidence"] = 0.35
            record["note"] = "width_blob" if blob else "not_two_straight_lines"
            unresolved.append(record)
    _junctions_on_pipe_bodies(pipes, junctions, cfg.node_cluster_px, take)
    return {
        "endpoints": endpoints,
        "junctions": junctions,
        "visual_crossings": crossings,
        "unresolved_crossings": unresolved,
    }


def _junctions_on_pipe_bodies(pipes: list[Branch], junctions: list[dict], radius: float, take) -> None:
    """A branch that meets the middle of another pipe is a junction even if the trunk was not split."""
    for branch in pipes:
        for endpoint in (branch.start, branch.end):
            if any(dist(endpoint, (node["xy"][0], node["xy"][1])) <= radius for node in junctions):
                continue
            for other in pipes:
                if other.id == branch.id or len(other.pixels) < 2:
                    continue
                gap, t = point_to_segment(
                    endpoint[0], endpoint[1], other.start[0], other.start[1], other.end[0], other.end[1]
                )
                if gap > radius or not (0.04 < t < 0.96):
                    continue
                proj = (
                    other.start[0] + t * (other.end[0] - other.start[0]),
                    other.start[1] + t * (other.end[1] - other.start[1]),
                )
                if dist(proj, other.start) <= radius or dist(proj, other.end) <= radius:
                    continue
                if angle_diff_deg(branch.orientation, other.orientation) < 18.0:
                    continue
                junctions.append(
                    {
                        "id": take("J"),
                        "xy": [round(float(endpoint[0]), 2), round(float(endpoint[1]), 2)],
                        "degree": 3,
                        "branch_ids": [branch.id, other.id],
                        "width_ratio": 1.0,
                        "angles": [round(branch.orientation, 2), round(other.orientation, 2)],
                        "kind": "junction",
                        "confidence": 0.7,
                        "note": "branch_meets_pipe_body",
                    }
                )
                break


def mark_port_endpoints(topology: dict, connections: list[Connection], radius: float = 8.0) -> None:
    ports = [connection.port_xy for connection in connections if connection.status == "verified"]
    for endpoint in topology["endpoints"]:
        xy = endpoint["xy"]
        if any(dist((xy[0], xy[1]), port) <= radius for port in ports):
            endpoint["role"] = "instrument_port"
