"""Verify instrument ports against pipe geometry. Proximity is not enough."""

from __future__ import annotations

import numpy as np

from layer_sep.config import LayerConfig, ResolvedThresholds
from layer_sep.geometry import clamp, directed_angle, directed_diff, dist
from layer_sep.types import Branch, Connection, Instrument


def _closest_contour(contour: np.ndarray, point: tuple[float, float]) -> tuple[tuple[float, float], float]:
    pts = contour.reshape(-1, 2).astype(np.float32)
    delta = pts - np.array(point, dtype=np.float32)
    distances = np.hypot(delta[:, 0], delta[:, 1])
    index = int(np.argmin(distances))
    return (float(pts[index, 0]), float(pts[index, 1])), float(distances[index])


def _approach_deg(branch: Branch, at_start: bool, target: tuple[float, float]) -> float:
    pts = branch.pixels if at_start else list(reversed(branch.pixels))
    step = min(12, len(pts) - 1)
    if step < 1:
        return 180.0
    if dist(pts[0], target) < 0.5:
        return 0.0
    heading = directed_angle(pts[step], pts[0])
    toward = directed_angle(pts[0], target)
    return directed_diff(heading, toward)


def _mostly_inside(branch: Branch, contour: np.ndarray, shape: tuple[int, int]) -> bool:
    mask = np.zeros(shape, np.uint8)
    cv2_draw(mask, contour)
    height, width = shape
    hits = 0
    for x, y in branch.pixels:
        if 0 <= x < width and 0 <= y < height and mask[y, x] > 0:
            hits += 1
    return hits / max(len(branch.pixels), 1) > 0.55


def cv2_draw(mask: np.ndarray, contour: np.ndarray) -> None:
    import cv2

    cv2.drawContours(mask, [contour], -1, 255, thickness=-1)


def promote_weak_instruments(
    instruments: list[Instrument],
    branches: list[Branch],
    resolved: ResolvedThresholds,
    cfg: LayerConfig,
    shape: tuple[int, int],
) -> bool:
    """Accept an empty rectangle only when a pipe outside it actually aims at it."""
    pipes = [branch for branch in branches if branch.label in {"thick_pipe", "thin_pipe"}]
    changed = False
    for instrument in instruments:
        if instrument.status != "weak" or instrument.in_grid:
            continue
        if _aimed_pipe(instrument, pipes, resolved, cfg, shape) is not None:
            instrument.status = "accepted"
            instrument.reasons.append("promoted_by_aimed_pipe")
            changed = True
    return changed


def _aimed_pipe(
    instrument: Instrument,
    pipes: list[Branch],
    resolved: ResolvedThresholds,
    cfg: LayerConfig,
    shape: tuple[int, int],
) -> Branch | None:
    best: Branch | None = None
    best_dist = 1e9
    for branch in pipes:
        if _mostly_inside(branch, instrument.contour, shape):
            continue
        for at_start, endpoint in ((True, branch.start), (False, branch.end)):
            target, gap = _closest_contour(instrument.contour, endpoint)
            reach = max(resolved.port_reach(branch.width), cfg.port_endpoint_min_px)
            if gap > reach or gap >= best_dist:
                continue
            if _approach_deg(branch, at_start, target) > cfg.port_angle_max_deg:
                continue
            best = branch
            best_dist = gap
    return best


def associate_ports(
    instruments: list[Instrument],
    branches: list[Branch],
    resolved: ResolvedThresholds,
    cfg: LayerConfig,
) -> list[Connection]:
    connections: list[Connection] = []
    serial = 1
    pipes = [
        branch
        for branch in branches
        if branch.label in {"thick_pipe", "thin_pipe", "ambiguous"} and branch.inside_instrument is None
    ]
    for instrument in instruments:
        if instrument.status != "accepted":
            continue
        best: dict[str, Connection] = {}
        for branch in pipes:
            candidate = _score_branch(instrument, branch, resolved, cfg, serial)
            if candidate is None:
                continue
            serial += 1
            current = best.get(branch.id)
            if current is None or _better(candidate, current):
                best[branch.id] = candidate
        connections.extend(best.values())
    for index, connection in enumerate(connections, start=1):
        connection.id = f"C-{index:04d}"
    return connections


def _better(new: Connection, old: Connection) -> bool:
    rank = {"verified": 2, "unverified": 1, "rejected": 0}
    if rank[new.status] != rank[old.status]:
        return rank[new.status] > rank[old.status]
    return new.confidence > old.confidence


def _score_branch(
    instrument: Instrument,
    branch: Branch,
    resolved: ResolvedThresholds,
    cfg: LayerConfig,
    serial: int,
) -> Connection | None:
    options = []
    for at_start, endpoint in ((True, branch.start), (False, branch.end)):
        target, gap = _closest_contour(instrument.contour, endpoint)
        options.append((gap, at_start, endpoint, target))
    options.sort(key=lambda item: item[0])
    gap, at_start, endpoint, target = options[0]
    reach = max(resolved.port_reach(branch.width), cfg.port_endpoint_min_px)
    if gap > reach:
        return None
    angle = _approach_deg(branch, at_start, target)
    thickness = None
    if branch.label == "thick_pipe":
        thickness = "thick"
    elif branch.label == "thin_pipe":
        thickness = "thin"
    end_score = clamp(1.0 - gap / max(reach, 1.0))
    angle_score = clamp(1.0 - angle / max(cfg.port_angle_max_deg, 1.0))
    class_score = branch.confidence if thickness else 0.0
    confidence = 0.42 * end_score + 0.33 * angle_score + 0.25 * class_score
    evidence = {
        "endpoint_distance": round(gap, 3),
        "angle_diff_deg": round(angle, 3),
        "reach_px": round(reach, 3),
        "width_outside": round(branch.width, 3),
        "branch_label": branch.label,
        "endpoint": [int(endpoint[0]), int(endpoint[1])],
        "boundary_point": [round(target[0], 2), round(target[1], 2)],
        "rejected_reason": None,
    }
    if angle > cfg.port_angle_max_deg:
        evidence["rejected_reason"] = "orientation_not_aimed"
        status = "rejected"
        confidence = min(confidence, 0.35)
    elif thickness is None:
        evidence["rejected_reason"] = "branch_not_a_confirmed_pipe"
        status = "unverified"
    elif confidence < cfg.connection_min_confidence:
        evidence["rejected_reason"] = "confidence_below_threshold"
        status = "unverified"
    else:
        status = "verified"
    return Connection(
        id=f"C-{serial:04d}",
        instrument_id=instrument.id,
        branch_id=branch.id,
        status=status,
        thickness_class=thickness,
        confidence=float(confidence),
        port_xy=(float(endpoint[0]), float(endpoint[1])),
        evidence=evidence,
    )


def connection_graph(
    instruments: list[Instrument],
    branches: list[Branch],
    connections: list[Connection],
) -> dict:
    nodes = []
    for instrument in instruments:
        if instrument.status != "accepted":
            continue
        nodes.append(
            {
                "id": instrument.id,
                "kind": "instrument",
                "shape": instrument.shape,
                "center": [round(instrument.center[0], 2), round(instrument.center[1], 2)],
            }
        )
    pipe_ids = {branch.id for branch in branches if branch.label in {"thick_pipe", "thin_pipe"}}
    for branch in branches:
        if branch.id not in pipe_ids:
            continue
        nodes.append(
            {
                "id": branch.id,
                "kind": "pipe",
                "thickness_class": "thick" if branch.label == "thick_pipe" else "thin",
                "confidence": round(branch.confidence, 4),
            }
        )
    edges = []
    for connection in connections:
        if connection.status != "verified":
            continue
        edges.append(
            {
                "id": connection.id,
                "source": connection.instrument_id,
                "target": connection.branch_id,
                "thickness_class": connection.thickness_class,
                "confidence": round(connection.confidence, 4),
                "port_xy": [round(connection.port_xy[0], 2), round(connection.port_xy[1], 2)],
            }
        )
    return {"nodes": nodes, "edges": edges, "edge_rule": "verified_connections_only"}
