"""Run layer separation without changing the source image or another pipeline."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import cv2
import numpy as np

from layer_sep.assign import assign_layers
from layer_sep.associate import associate_ports, connection_graph, promote_weak_instruments
from layer_sep.config import LayerConfig
from layer_sep.instruments import detect_instruments, raster_instruments
from layer_sep.junctions import detect_junctions, mark_port_endpoints
from layer_sep.ocr_stage import read_text
from layer_sep.overlay import (
    draw_text_boxes,
    save_classes,
    save_comparison,
    save_gray,
    save_instruments,
    save_mask,
    save_overlay,
    save_width,
)
from layer_sep.preprocess import map_mask_to_original, map_point_to_original, preprocess
from layer_sep.skeleton import skeleton_image, trace_mask
from layer_sep.strokes import classify_paths, reconstruct_mask
from layer_sep.svg_prep import prepare_thick_svg
from layer_sep.text_assoc import associate_text, link_tags, raster_text
from layer_sep.types import Branch, Connection, Instrument, LayerResult, TextItem
from layer_sep.validate import build_report


def load_bgr(path: Path) -> np.ndarray:
    data = np.fromfile(str(path), dtype=np.uint8)
    image = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(path)
    return image


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _classify(binary, width_map, paths, mask, cfg, shape):
    return classify_paths(paths, binary, width_map, mask, cfg, shape)


def _map_point(point: tuple[float, float] | list[float], prepared) -> tuple[float, float]:
    return map_point_to_original((float(point[0]), float(point[1])), prepared)


def _to_original_space(
    prepared,
    branches: list[Branch],
    instruments: list[Instrument],
    connections: list[Connection],
    junctions: dict,
    masks: dict[str, np.ndarray],
) -> dict[str, np.ndarray]:
    if prepared.to_original is None:
        return masks
    height, width = prepared.working_shape
    for branch in branches:
        mapped = []
        for x, y in branch.pixels:
            ox, oy = _map_point((x, y), prepared)
            ix, iy = int(round(ox)), int(round(oy))
            if 0 <= ix < width and 0 <= iy < height:
                mapped.append((ix, iy))
        if len(mapped) >= 2:
            branch.pixels = mapped
    for instrument in instruments:
        pts = []
        for point in instrument.contour.reshape(-1, 2):
            ox, oy = _map_point((float(point[0]), float(point[1])), prepared)
            pts.append([int(round(ox)), int(round(oy))])
        contour = np.array(pts, np.int32).reshape(-1, 1, 2)
        x, y, w, h = cv2.boundingRect(contour)
        instrument.contour = contour
        instrument.bbox = (int(x), int(y), int(w), int(h))
        instrument.center = (x + w / 2.0, y + h / 2.0)
    for connection in connections:
        ox, oy = _map_point(connection.port_xy, prepared)
        connection.port_xy = (ox, oy)
    for group in ("endpoints", "junctions", "visual_crossings", "unresolved_crossings"):
        for node in junctions.get(group, []):
            ox, oy = _map_point(node["xy"], prepared)
            node["xy"] = [round(ox, 2), round(oy, 2)]
    return {name: map_mask_to_original(mask, prepared) for name, mask in masks.items()}


def separate_layers(
    image_path: str | Path,
    out_dir: str | Path,
    cfg: LayerConfig | None = None,
    skip_ocr: bool = False,
    texts_override: list[TextItem] | None = None,
    save_debug: bool = True,
) -> LayerResult:
    """Separate one drawing. Writes a new directory and does not edit the source."""
    cfg = cfg or LayerConfig()
    image_path = Path(image_path)
    before = _sha256(image_path)
    original = load_bgr(image_path)
    height, width = original.shape[:2]
    dest = Path(out_dir) / image_path.stem
    debug = dest / "debug"
    if save_debug:
        debug.mkdir(parents=True, exist_ok=True)
        save_mask(debug / "01_original.png", original)

    prepared = preprocess(original, cfg)
    if save_debug:
        save_gray(debug / "02_gray.png", prepared.gray)
        save_mask(debug / "03_binary.png", prepared.binary)
        save_width(debug / "04_width.png", prepared.width_map, prepared.binary)

    paths = trace_mask(prepared.binary, cfg.corner_deg)
    skeleton = skeleton_image(prepared.binary)
    if save_debug:
        save_mask(debug / "05_skeleton.png", skeleton)

    # Thresholds are provisional until the first width sample exists.
    from layer_sep.config import resolve_thresholds

    provisional = resolve_thresholds(cfg, 2.0, None, prepared.binary.shape[:2])
    instruments = detect_instruments(prepared.binary, skeleton, cfg, provisional)
    instrument_mask = raster_instruments(prepared.binary.shape[:2], instruments)
    branches, resolved, text_geometry, _segments = _classify(
        prepared.binary,
        prepared.width_map,
        paths,
        instrument_mask,
        cfg,
        prepared.binary.shape[:2],
    )
    if promote_weak_instruments(instruments, branches, resolved, cfg, prepared.binary.shape[:2]):
        instrument_mask = raster_instruments(prepared.binary.shape[:2], instruments)
        branches, resolved, text_geometry, _segments = _classify(
            prepared.binary,
            prepared.width_map,
            paths,
            instrument_mask,
            cfg,
            prepared.binary.shape[:2],
        )
    connections = associate_ports(instruments, branches, resolved, cfg)
    junctions = detect_junctions(branches, prepared.width_map, cfg)
    mark_port_endpoints(junctions, connections)

    warnings: list[str] = []
    if resolved.thick_peak is None:
        warnings.append(
            "No separate thick-stroke mode was measured. Strokes were not promoted to thick pipes by width alone."
        )
    if resolved.thin_peak is None:
        warnings.append("No stable stroke-width mode was measured. Pipe labels were withheld.")

    shape = prepared.binary.shape[:2]
    exclude = instrument_mask
    masks = {
        "thick": reconstruct_mask(shape, branches, {"thick_pipe"}, prepared.binary, exclude),
        "thin": reconstruct_mask(shape, branches, {"thin_pipe"}, prepared.binary, exclude),
        "ambiguous": reconstruct_mask(shape, branches, {"ambiguous"}, prepared.binary, None),
        "instrument": instrument_mask,
        "text_geometry": text_geometry,
        "frame": reconstruct_mask(shape, branches, {"frame"}, prepared.binary, None),
        "binary": prepared.binary,
    }
    masks = _to_original_space(prepared, branches, instruments, connections, junctions, masks)

    ocr_warning = None
    if texts_override is not None:
        texts = texts_override
    elif skip_ocr:
        texts = []
        ocr_warning = "OCR was skipped."
    else:
        texts, ocr_warning = read_text(original, cfg)
    link_tags(texts, instruments)
    associate_text(texts, branches, cfg, resolved)
    text_mask = raster_text((height, width), masks["text_geometry"], texts)
    assignments = assign_layers(instruments, branches, connections, texts)
    graph = connection_graph(instruments, branches, connections)

    if save_debug:
        save_classes(
            debug / "06_stroke_classes.png",
            (height, width),
            {
                "thick": masks["thick"],
                "thin": masks["thin"],
                "ambiguous": masks["ambiguous"],
                "text": text_mask,
                "instrument": masks["instrument"],
                "frame": masks["frame"],
            },
        )
        save_instruments(debug / "07_instruments.png", original, instruments)
        draw_text_boxes(debug / "08_text.png", original, texts)
        _write_json(debug / "branches.json", {"branches": [branch.summary() for branch in branches]})
        _write_json(
            debug / "instruments.json",
            {"instruments": [item.summary() for item in instruments]},
        )

    dest.mkdir(parents=True, exist_ok=True)
    save_mask(dest / "thick_pipe_mask.png", masks["thick"])
    save_mask(dest / "thin_pipe_mask.png", masks["thin"])
    save_mask(dest / "instrument_mask.png", masks["instrument"])
    save_mask(dest / "text_mask.png", text_mask)
    _write_json(dest / "junctions.json", junctions)
    _write_json(
        dest / "connections.json",
        {
            "graph": graph,
            "verified": [item.summary() for item in connections if item.status == "verified"],
            "unverified": [item.summary() for item in connections if item.status == "unverified"],
            "rejected": [item.summary() for item in connections if item.status == "rejected"],
        },
    )
    _write_json(dest / "layer_assignments.json", assignments)
    overlay = save_overlay(
        dest / "debug_overlay.png",
        original,
        masks["thick"],
        masks["thin"],
        masks["instrument"],
        text_mask,
        junctions,
        connections,
        ambiguous=masks["ambiguous"],
    )
    save_comparison(dest / "comparison.png", original, overlay)
    prepare_thick_svg(masks["thick"], branches, cfg, dest / "svg_prep")

    after = _sha256(image_path)
    report = build_report(
        image_path=str(image_path),
        width=width,
        height=height,
        original_sha256=before,
        original_intact=before == after,
        branches=branches,
        instruments=instruments,
        connections=connections,
        texts=texts,
        junctions=junctions,
        binary=masks["binary"],
        thick_mask=masks["thick"],
        thin_mask=masks["thin"],
        warnings=warnings,
        ocr_warning=ocr_warning,
        skew_deg=prepared.skew_deg,
        thin_peak=resolved.thin_peak,
        thick_peak=resolved.thick_peak,
    )
    _write_json(dest / "validation_report.json", report)
    return LayerResult(
        image_path=str(image_path),
        width=width,
        height=height,
        original_sha256=before,
        branches=branches,
        instruments=instruments,
        texts=texts,
        connections=connections,
        junctions=junctions,
        assignments=assignments,
        validation=report,
        output_dir=str(dest),
        warnings=report["warnings"],
    )
