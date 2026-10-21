# P&ID layer separation

Rule-based separation of a P&ID into thick pipes, thin pipes, instruments, text, and junctions. The module reads the original image and writes a new folder. It does not modify that image and it does not replace the graph builder in another pipeline.

No object detector is used. Stroke class comes from the distance transform, directional openings, and line geometry. PaddleOCR supplies text and tags. Instruments without readable text are still detected from their contours.

## Data flow

1. Load the drawing at full resolution.
2. Build an ink mask, estimate skew, and compute a local stroke width (`2 * distance transform`).
3. Skeletonize and split runs at junctions and corners.
4. Detect circles, ellipses, rectangles, diamonds, and polygons. Empty rectangles are accepted only when a pipe outside the shape aims at them.
5. Estimate thick and thin width modes from long, straight runs. The valley between those modes stays ambiguous.
6. Keep a run as a pipe only when width, length, linearity, and continuity agree. Text strokes and short symbol strokes stay out.
7. For each instrument, pair a port with a pipe only when the pipe endpoint is close, aimed at the boundary, and continuous. A line that merely passes nearby is rejected.
8. Assign layers from verified links: thick only, thin only, both (stored on the thick layer), or review.
9. Associate diameter and pressure text only when one pipe is clearly better than the next. Otherwise the association stays unresolved.
10. Write masks, the connection graph, junction records, an overlay, and a thick-layer SVG preparation folder. Cleanup is saved as a second mask.

Accuracy is not reported as a score unless ground-truth masks are passed in. The validation file says so explicitly.

## Run

```text
python run_layers.py path\to\diagram.png --out output\layer_separation
python run_layers.py path\to\diagram.png --skip-ocr
python -m unittest tests.test_layers
```

Outputs for each drawing:

- `thick_pipe_mask.png`
- `thin_pipe_mask.png`
- `instrument_mask.png`
- `text_mask.png`
- `junctions.json`
- `connections.json`
- `layer_assignments.json`
- `debug_overlay.png`
- `validation_report.json`
- `comparison.png`
- `debug/` stage images
- `svg_prep/` raw mask, cleaned copy, centerlines, and contours in the original pixel coordinates

Thresholds live in `layer_sep/config.py`. Lengths scale with the measured thin-stroke width.
# layers
