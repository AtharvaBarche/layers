"""Layer ownership from verified connections only."""

from __future__ import annotations

from layer_sep.types import Branch, Connection, Instrument, TextItem


def assign_layers(
    instruments: list[Instrument],
    branches: list[Branch],
    connections: list[Connection],
    texts: list[TextItem],
) -> dict:
    by_instrument: dict[str, list[Connection]] = {}
    for connection in connections:
        by_instrument.setdefault(connection.instrument_id, []).append(connection)
    instrument_rows = []
    for instrument in instruments:
        if instrument.status != "accepted":
            instrument_rows.append(
                {
                    "id": instrument.id,
                    "layer": "review",
                    "reason": "not_accepted",
                    "shape": instrument.shape,
                    "confidence": round(instrument.confidence, 4),
                    "verified_thick": 0,
                    "verified_thin": 0,
                    "unverified": 0,
                    "bbox": instrument.summary()["bbox"],
                    "ocr_ids": list(instrument.ocr_ids),
                }
            )
            continue
        rows = by_instrument.get(instrument.id, [])
        verified = [row for row in rows if row.status == "verified"]
        unverified = [row for row in rows if row.status == "unverified"]
        thick = sum(1 for row in verified if row.thickness_class == "thick")
        thin = sum(1 for row in verified if row.thickness_class == "thin")
        if unverified:
            layer, reason = "review", "unverified_connection"
        elif thick and thin:
            layer, reason = "thick", "both_networks"
        elif thick:
            layer, reason = "thick", "exclusive_thick"
        elif thin:
            layer, reason = "thin", "exclusive_thin"
        else:
            layer, reason = "review", "no_verified_connection"
        instrument_rows.append(
            {
                "id": instrument.id,
                "layer": layer,
                "reason": reason,
                "shape": instrument.shape,
                "confidence": round(instrument.confidence, 4),
                "verified_thick": thick,
                "verified_thin": thin,
                "unverified": len(unverified),
                "bbox": instrument.summary()["bbox"],
                "center": [round(instrument.center[0], 2), round(instrument.center[1], 2)],
                "ocr_ids": list(instrument.ocr_ids),
            }
        )
    pipe_rows = []
    for branch in branches:
        if branch.label not in {"thick_pipe", "thin_pipe", "ambiguous"}:
            continue
        if branch.label == "thick_pipe":
            layer = "thick"
        elif branch.label == "thin_pipe":
            layer = "thin"
        else:
            layer = "review"
        pipe_rows.append(
            {
                "id": branch.id,
                "layer": layer,
                "label": branch.label,
                "width": round(branch.width, 3),
                "length": round(branch.length, 2),
                "confidence": round(branch.confidence, 4),
                "reasons": list(branch.reasons),
            }
        )
    text_rows = [text.summary() for text in texts]
    return {
        "instruments": instrument_rows,
        "pipes": pipe_rows,
        "text": text_rows,
        "rules": {
            "exclusive_thick": "thick",
            "exclusive_thin": "thin",
            "both_networks": "thick",
            "unverified_or_ambiguous": "review",
        },
    }
