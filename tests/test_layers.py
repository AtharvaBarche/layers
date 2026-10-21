"""Synthetic checks for crossings, branches, overlap, and mixed connections.

Pixel scores and connection scores are computed separately. They describe
this generated drawing only.
"""

from __future__ import annotations

import math
import unittest
from pathlib import Path

import cv2
import numpy as np

from layer_sep.config import LayerConfig
from layer_sep.evaluate import connection_scores, pixel_scores
from layer_sep.geometry import axis_deviation
from layer_sep.junctions import is_visual_crossing
from layer_sep.pipeline import separate_layers
from layer_sep.strokes import width_class
from tests.synthetic import (
    CROSS_X,
    DISTANT_CIRCLE,
    GAP_CIRCLE,
    HEIGHT,
    RECT,
    SCRAP,
    THICK_CIRCLE,
    THICK_Y,
    THIN_BRANCH_X,
    THIN_CIRCLE,
    THIN_Y,
    WIDTH,
    build_scene,
)

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "output" / "layer_separation"


def _near(nodes: list[dict], xy: tuple[float, float], tol: float = 18.0) -> dict | None:
    best = None
    best_d = tol
    for node in nodes:
        d = math.hypot(node["xy"][0] - xy[0], node["xy"][1] - xy[1])
        if d <= best_d:
            best = node
            best_d = d
    return best


class RuleTests(unittest.TestCase):
    def test_width_valley_stays_ambiguous(self) -> None:
        kind, _score = width_class(6.0, 3.0, 9.0, LayerConfig())
        self.assertEqual(kind, "ambiguous")
        thick, _ = width_class(9.0, 3.0, 9.0, LayerConfig())
        thin, _ = width_class(3.0, 3.0, 9.0, LayerConfig())
        self.assertEqual(thick, "thick")
        self.assertEqual(thin, "thin")

    def test_crossing_angles_are_not_a_junction_rule(self) -> None:
        self.assertTrue(is_visual_crossing([0.0, 180.0, 90.0, 270.0], 22.0))
        self.assertFalse(is_visual_crossing([0.0, 90.0, 200.0], 22.0))

    def test_axis_deviation_of_a_near_horizontal_line(self) -> None:
        self.assertAlmostEqual(axis_deviation(3.0), 3.0, places=3)
        self.assertAlmostEqual(axis_deviation(88.0), -2.0, places=3)

    def test_pixel_and_connection_metrics_are_independent(self) -> None:
        pred = np.zeros((6, 6), np.uint8)
        truth = np.zeros((6, 6), np.uint8)
        pred[1:3, 1:5] = 255
        truth[1:3, 1:5] = 255
        pixels = pixel_scores(pred, truth)
        links = connection_scores([(0, 0, "thick")], [(50, 50, "thick")], tolerance_px=5)
        self.assertGreater(pixels["iou"], 0.99)
        self.assertEqual(links["tp"], 0)
        self.assertEqual(links["fn"], 1)


class SceneTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.scene = build_scene(OUT / "_fixture")
        cls.result = separate_layers(
            cls.scene.path,
            OUT,
            cfg=LayerConfig(deskew=False),
            skip_ocr=True,
            texts_override=cls.scene.texts,
        )
        dest = Path(cls.result.output_dir)
        cls.thick = cv2.imread(str(dest / "thick_pipe_mask.png"), cv2.IMREAD_GRAYSCALE)
        cls.thin = cv2.imread(str(dest / "thin_pipe_mask.png"), cv2.IMREAD_GRAYSCALE)
        cls.instruments = cv2.imread(str(dest / "instrument_mask.png"), cv2.IMREAD_GRAYSCALE)
        cls.text = cv2.imread(str(dest / "text_mask.png"), cv2.IMREAD_GRAYSCALE)

    def test_required_files_and_original_coordinates(self) -> None:
        dest = Path(self.result.output_dir)
        for name in (
            "thick_pipe_mask.png",
            "thin_pipe_mask.png",
            "instrument_mask.png",
            "text_mask.png",
            "junctions.json",
            "connections.json",
            "layer_assignments.json",
            "debug_overlay.png",
            "validation_report.json",
        ):
            self.assertTrue((dest / name).is_file(), name)
        self.assertEqual(self.thick.shape, (HEIGHT, WIDTH))
        self.assertEqual(self.thin.shape, (HEIGHT, WIDTH))
        self.assertTrue(self.result.validation["image"]["original_unchanged"])
        self.assertIsNone(self.result.validation["accuracy"]["pixel_iou"])
        self.assertIn("not claimed", self.result.validation["accuracy"]["note"])
        raw = dest / "svg_prep" / "thick_pipe_mask_raw.png"
        cleaned = dest / "svg_prep" / "thick_pipe_mask_cleaned.png"
        self.assertTrue(raw.is_file())
        self.assertTrue(cleaned.is_file())

    def test_known_pipe_pixels_and_rejected_strokes(self) -> None:
        self.assertGreater(self.thick[THICK_Y, 100], 0)
        self.assertGreater(self.thin[THIN_Y, 100], 0)
        self.assertEqual(int(self.thick[SCRAP[0][1], 50]), 0)
        self.assertEqual(int(self.thick[AMBIG_Y := 700, 400]), 0)
        self.assertEqual(int(self.thin[700, 400]), 0)
        # OCR box overlaps the thick pipe. The pipe pixel stays.
        self.assertGreater(self.thick[THICK_Y, 700], 0)
        self.assertGreater(self.text[210, 690], 0)

    def test_crossing_and_branch(self) -> None:
        junctions = self.result.junctions
        crossing = _near(junctions["visual_crossings"], (CROSS_X, THICK_Y))
        self.assertIsNotNone(crossing, msg=f"crossings={junctions['visual_crossings']}")
        self.assertIsNone(_near(junctions["junctions"], (CROSS_X, THICK_Y), tol=10))
        branch = _near(junctions["junctions"], (THIN_BRANCH_X, THIN_Y))
        self.assertIsNotNone(branch, msg=f"junctions={junctions['junctions']}")
        thick_tee = _near(junctions["junctions"], (230, THICK_Y))
        self.assertIsNotNone(thick_tee, msg=f"junctions={junctions['junctions']}")

    def test_layer_rules_for_instruments(self) -> None:
        thick = self._layer_at(THICK_CIRCLE[:2])
        thin = self._layer_at(THIN_CIRCLE[:2])
        both = self._layer_at((RECT[0] + RECT[2] / 2, RECT[1] + RECT[3] / 2))
        distant = self._layer_at(DISTANT_CIRCLE[:2])
        gapped = self._layer_at(GAP_CIRCLE[:2])
        self.assertEqual(thick["reason"], "exclusive_thick", thick)
        self.assertEqual(thin["reason"], "exclusive_thin", thin)
        self.assertEqual(both["layer"], "thick", both)
        self.assertEqual(both["reason"], "both_networks", both)
        self.assertEqual(distant["layer"], "review", distant)
        self.assertEqual(gapped["layer"], "review", gapped)
        self.assertEqual(gapped["verified_thick"], 0)

    def test_overlapping_symbols_do_not_erase_the_pipe(self) -> None:
        accepted = [item for item in self.result.instruments if item.status == "accepted"]
        near_overlap = [
            item
            for item in accepted
            if math.hypot(item.center[0] - 900, item.center[1] - THICK_Y) < 80
        ]
        self.assertGreaterEqual(len(near_overlap), 1)
        self.assertGreater(self.instruments[THICK_Y, 900], 0)
        self.assertGreater(self.thick[THICK_Y, 820], 0)
        self.assertGreater(self.thick[THICK_Y, 1040], 0)

    def test_text_associations_are_not_invented(self) -> None:
        by_text = {item.text: item for item in self.result.texts}
        self.assertEqual(by_text["DN150"].kind, "DIAMETER")
        self.assertEqual(by_text["DN150"].association_status, "assigned")
        self.assertEqual(by_text["6 bar"].kind, "PRESSURE")
        self.assertEqual(by_text["6 bar"].association_status, "assigned")
        self.assertEqual(by_text["FT-101"].association_status, "instrument_tag")
        self.assertIsNone(by_text["FT-101"].pipe_id)
        self.assertEqual(by_text["SEE NOTE"].association_status, "unresolved")
        self.assertIsNone(by_text["SEE NOTE"].pipe_id)

    def test_synthetic_measurements_are_reported_separately(self) -> None:
        pixels = pixel_scores(self.thick, self.scene.thick_gt)
        truth_links = [
            (float(THICK_CIRCLE[0]), float(THICK_CIRCLE[1]), "thick"),
            (float(THIN_CIRCLE[0]), float(THIN_CIRCLE[1]), "thin"),
            (float(RECT[0] + RECT[2] / 2), float(RECT[1] + RECT[3] / 2), "thick"),
            (float(RECT[0] + RECT[2] / 2), float(RECT[1] + RECT[3] / 2), "thin"),
        ]
        predicted = []
        centers = {row["id"]: row["center"] for row in self.result.assignments["instruments"]}
        for connection in self.result.connections:
            if connection.status != "verified" or connection.instrument_id not in centers:
                continue
            cx, cy = centers[connection.instrument_id]
            predicted.append((cx, cy, connection.thickness_class))
        links = connection_scores(predicted, truth_links, tolerance_px=30)
        self.assertGreater(pixels["iou"], 0.35, msg=pixels)
        self.assertGreater(links["recall"], 0.99, msg=links)
        self.assertIsNone(self.result.validation["accuracy"]["connection_f1"])

    def _layer_at(self, xy: tuple[float, float]) -> dict:
        best = None
        best_d = 36.0
        for row in self.result.assignments["instruments"]:
            if row["layer"] == "review" and row["reason"] == "not_accepted":
                continue
            center = row.get("center")
            if center is None:
                continue
            d = math.hypot(center[0] - xy[0], center[1] - xy[1])
            if d < best_d:
                best = row
                best_d = d
        self.assertIsNotNone(best, msg=f"no instrument near {xy}")
        return best


if __name__ == "__main__":
    unittest.main()
