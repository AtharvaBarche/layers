"""In-memory records. Masks stay in numpy arrays; JSON gets summaries."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass
class Branch:
    id: str
    pixels: list[tuple[int, int]]
    component_id: str
    width: float = 0.0
    width_std: float = 0.0
    length: float = 0.0
    orientation: float = 0.0
    linearity: float = 0.0
    label: str = "unassigned"
    confidence: float = 0.0
    reasons: list[str] = field(default_factory=list)
    inside_instrument: str | None = None

    @property
    def start(self) -> tuple[int, int]:
        return self.pixels[0]

    @property
    def end(self) -> tuple[int, int]:
        return self.pixels[-1]

    def summary(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "label": self.label,
            "confidence": round(self.confidence, 4),
            "width": round(self.width, 3),
            "width_std": round(self.width_std, 3),
            "length": round(self.length, 2),
            "orientation": round(self.orientation, 2),
            "linearity": round(self.linearity, 3),
            "start": list(self.start),
            "end": list(self.end),
            "component_id": self.component_id,
            "inside_instrument": self.inside_instrument,
            "reasons": list(self.reasons),
            "pixel_count": len(self.pixels),
        }


@dataclass
class Instrument:
    id: str
    shape: str
    bbox: tuple[int, int, int, int]
    center: tuple[float, float]
    confidence: float
    contour: np.ndarray
    status: str
    internal_ink: float = 0.0
    in_grid: bool = False
    child_count: int = 0
    ocr_ids: list[str] = field(default_factory=list)
    reasons: list[str] = field(default_factory=list)

    def summary(self) -> dict[str, Any]:
        x, y, w, h = self.bbox
        return {
            "id": self.id,
            "shape": self.shape,
            "bbox": [x, y, x + w, y + h],
            "center": [round(self.center[0], 2), round(self.center[1], 2)],
            "confidence": round(self.confidence, 4),
            "status": self.status,
            "internal_ink": round(self.internal_ink, 4),
            "in_grid": self.in_grid,
            "child_count": self.child_count,
            "ocr_ids": list(self.ocr_ids),
            "reasons": list(self.reasons),
        }


@dataclass
class TextItem:
    id: str
    text: str
    bbox: tuple[float, float, float, float]
    confidence: float
    kind: str
    angle: float
    source: str
    pipe_id: str | None = None
    instrument_id: str | None = None
    association_status: str = "unresolved"
    association_confidence: float = 0.0
    association_reason: str = ""

    @property
    def center(self) -> tuple[float, float]:
        x1, y1, x2, y2 = self.bbox
        return ((x1 + x2) * 0.5, (y1 + y2) * 0.5)

    def summary(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "text": self.text,
            "bbox": [round(v, 2) for v in self.bbox],
            "confidence": round(self.confidence, 4),
            "kind": self.kind,
            "angle": round(self.angle, 2),
            "source": self.source,
            "pipe_id": self.pipe_id,
            "instrument_id": self.instrument_id,
            "association_status": self.association_status,
            "association_confidence": round(self.association_confidence, 4),
            "association_reason": self.association_reason,
        }


@dataclass
class Connection:
    id: str
    instrument_id: str
    branch_id: str
    status: str
    thickness_class: str | None
    confidence: float
    port_xy: tuple[float, float]
    evidence: dict[str, Any]

    def summary(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "instrument_id": self.instrument_id,
            "branch_id": self.branch_id,
            "status": self.status,
            "thickness_class": self.thickness_class,
            "confidence": round(self.confidence, 4),
            "port_xy": [round(self.port_xy[0], 2), round(self.port_xy[1], 2)],
            "evidence": self.evidence,
        }


@dataclass
class LayerResult:
    image_path: str
    width: int
    height: int
    original_sha256: str
    branches: list[Branch]
    instruments: list[Instrument]
    texts: list[TextItem]
    connections: list[Connection]
    junctions: dict[str, Any]
    assignments: dict[str, Any]
    validation: dict[str, Any]
    output_dir: str
    warnings: list[str] = field(default_factory=list)
