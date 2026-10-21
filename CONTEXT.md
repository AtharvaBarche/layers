# Layer separation context

This folder (`imagelayering`) is an independent rule-based split of a P&ID into thick pipes, thin pipes, instruments, text, and junctions. `separate_layers` in `layer_sep/pipeline.py` reads the source image, checksums it before and after, and writes a new directory. The original file is never edited.

Two nearby projects were inspected and must stay untouched:

- `C:\Users\HP\Desktop\ossr_p&id` — its own rule-based `process_image` (`pipeline.py`): preprocess, width, pipes, symbols, OCR, then symbol graphs. Do not edit it and do not replace its graph builder.
- `C:\Users\HP\Desktop\P&ID_extraction` — YOLO symbol detection (`detect.py`) plus `graph.py` (`build_graph`). Do not edit it. Do not call YOLO from this folder.

## Module map

Public entry is `separate_layers` (`layer_sep/__init__.py`). The CLI is `run_layers.py` (`collect_images` → `separate_layers`).

Order inside `separate_layers`:

1. `load_bgr` reads the image. `preprocess` (`preprocess.py`) builds a working copy: grayscale, denoise, Otsu AND adaptive threshold, tiny-component cleanup, optional deskew, then stroke width as `2 ×` distance transform. Deskew runs only when the estimated angle is between `deskew_min_deg` (0.8) and `deskew_max_deg` (12).
2. `trace_mask` and `skeleton_image` (`skeleton.py`). Skeletonization uses `skimage.morphology.skeletonize`. Paths split at junctions and at corners (`corner_deg`).
3. Provisional `resolve_thresholds`, then `detect_instruments` and `raster_instruments` (`instruments.py`): contours, Hough circles that sit on an ink ring, and skeleton faces. Shapes scored are circles, ellipses, rectangles, diamonds, and polygons. Empty rectangles stay weak until `promote_weak_instruments` (`associate.py`) sees a pipe aimed at them. `_reject_header_rows` rejects a row of similar circles in the top of the sheet. OCR is not required for this step.
4. `classify_paths` (`strokes.py`): `find_width_peaks` on long straight runs, `resolve_thresholds` again, `label_branches`, then `bridge_collinear_pipes`. A run is `thick_pipe` or `thin_pipe` only when width class, length, linearity, and score all pass, and text overlap is under 0.55. Otherwise it stays `ambiguous`, `noise`, `text`, `frame`, or `instrument_stroke`.
5. `associate_ports` (`associate.py`). A candidate exists only when a pipe endpoint is within reach of the symbol boundary. Aim, confirmed thickness, and `connection_min_confidence` (0.58) decide `verified`, `unverified`, or `rejected`. A line that only passes nearby is not a connection.
6. `detect_junctions` and `mark_port_endpoints` (`junctions.py`). Degree 3 is a junction. Degree 4 with two collinear pairs and no width blob is a visual crossing, not a junction. Anything else at that degree is an unresolved crossing.
7. Masks are rebuilt with `reconstruct_mask` and mapped back with `map_mask_to_original` / `map_point_to_original` when deskew ran. Instrument pixels are excluded from the pipe masks. The text mask is separate and does not punch out pipe pixels.
8. Text: `read_text` (`ocr_stage.py`, PaddleOCR) unless `texts_override` or `--skip-ocr`. `link_tags` and `associate_text` (`text_assoc.py`) attach a diameter or pressure string only when one pipe beats the next by `text_assign_margin`. Otherwise the association stays `unresolved`.
9. `assign_layers` (`assign.py`) and `connection_graph` (`associate.py`). `prepare_thick_svg` (`svg_prep.py`) writes a raw thick mask and a cleaned copy. `build_report` (`validate.py`) writes the validation file. It sets accuracy to null unless a ground-truth dict is passed in. The pipeline does not pass one.

Thresholds live in `LayerConfig` (`layer_sep/config.py`). `resolve_thresholds` scales length, spur, text height, text distance, and symbol size from the measured thin-stroke peak (fallback scale 2.0). Width alone is not a pipe label. If the two peaks are too close, the thick peak is dropped and strokes are not promoted to thick by width.

## Layer ownership

From `assign_layers`, using verified connections only:

- Accepted instrument with only verified thick links: layer `thick`, reason `exclusive_thick`.
- Only verified thin links: layer `thin`, reason `exclusive_thin`.
- Verified thick and verified thin, and no unverified link: layer `thick`, reason `both_networks`.
- Any unverified link: layer `review`, reason `unverified_connection`. This check happens before the exclusive and both-network cases.
- No verified link: layer `review`, reason `no_verified_connection`.
- Instrument that was not accepted: layer `review`, reason `not_accepted`.

Pipe rows: `thick_pipe` → thick, `thin_pipe` → thin, `ambiguous` → review. `width_class` leaves the valley between the thin and thick modes as `ambiguous`. Do not force that band to a thickness.

## How to run

```text
python -m unittest tests.test_layers
python run_layers.py path\to\diagram.png --out output\layer_separation
python run_layers.py path\to\diagram.png --skip-ocr
python run_layers.py path\to\diagram.png --no-deskew
```

Each drawing is written to `output/layer_separation/<stem>/`. Required names:

- `thick_pipe_mask.png`, `thin_pipe_mask.png`, `instrument_mask.png`, `text_mask.png`
- `junctions.json`, `connections.json`, `layer_assignments.json`
- `debug_overlay.png`, `comparison.png`, `validation_report.json`
- `debug/` stage images plus `branches.json` and `instruments.json`
- `svg_prep/`: `thick_pipe_mask_raw.png`, `thick_pipe_mask_cleaned.png`, `thick_centerlines.json`, `thick_contours.json`, `manifest.json` (original pixel coordinates; RDP epsilon 1 px)

`tests/synthetic.py` `build_scene` draws a 1400×900 sheet: thick line at y=250 (width 9), thin line at y=560 (width 3), thick circle, thin circle, a rectangle tied to both, a diamond with an overlapping circle, a vertical thin line crossing both mains at x=760, a thin tee at x=430, a thick tee at x=230, a short thick scrap, a mid-width line at y=700, a distant circle, a circle with a stub that does not reach it, and the strings `DN150`, `6 bar`, `FT-101`, and `SEE NOTE`. `tests/test_layers.py` checks the width valley, crossings versus junctions, those layer reasons, pipe pixels under an overlapping symbol and under the `DN150` box, and that `SEE NOTE` stays unresolved.

## Measured status

### Synthetic scene only

Source of the saved report: `output/layer_separation/scene/validation_report.json`, image `output/layer_separation/_fixture/scene.png` (1400×900). OCR was not run; four injected text items were passed in. `original_unchanged` is true. Skew applied is 0. Width modes: thin peak 5.97, thick peak 11.804. Warnings: none.

Branch counts: thick_pipe 8, thin_pipe 10, ambiguous 8, noise 190, instrument_stroke 134, text 2. Instruments accepted 7 of 7 recorded. Text regions 4. Endpoints 16, junctions 4, visual crossings 2, unresolved crossings 0. Verified connections 6, unverified 1 (`C-0010`, ambiguous branch, reason `branch_not_a_confirmed_pipe`), rejected nearby 7. Thick pixels 14008, thin pixels 12095.

`accuracy.pixel_iou` and `connection_f1` in that report are null. The note says no manually verified ground truth was provided and high accuracy is not claimed. `test_synthetic_measurements_are_reported_separately` scores the thick mask against the generator mask and only asserts IoU greater than 0.35, plus connection recall greater than 0.99. It does not write the score. No output file stores a thick IoU or a thin IoU. Do not quote a specific IoU, and do not treat this drawing as a real-drawing accuracy result. Figures near 0.92 elsewhere in the JSON are widths or confidences, not IoU.

### Real drawing

`C:\Users\HP\Desktop\P&ID_extraction\data\images\pid_example.png` (1139×797) was run with PaddleOCR. Report: `output/layer_separation/pid_example/validation_report.json`. `original_unchanged` is true. Skew applied is 0. Width modes: thin peak 1.809, thick peak 4.064. Warnings: none (OCR was not skipped).

Branch counts: thick_pipe 24, thin_pipe 107, ambiguous 418, text 29, instrument_stroke 571, noise 8866. Instruments accepted 4 of 8 recorded. Text regions 59. Endpoints 187, junctions 5, visual crossings 0, unresolved crossings 0. Verified connections 0, unverified 8, rejected nearby 7. Thick pixels 10964, thin pixels 20226. Mean pipe confidence 0.8124, mean accepted-instrument confidence 0.8922, verified-connection mean null, text-association mean 0.7721. Seven unverified links are `branch_not_a_confirmed_pipe`; one thick link (`C-0005`) is `confidence_below_threshold`.

`accuracy.available` is false. `pixel_iou` and `connection_f1` are null. There is no manual ground truth. Do not claim high accuracy on this drawing.

## Limits to leave alone

Do not clear these by inventing labels:

- Short spurs (`length <= spur_max`) stay `noise`. Runs that miss the minimum length or `min_linearity` (0.70), or whose width sits in the ambiguous band, stay `ambiguous` and their pipe row is `review`.
- Header-row circles (three or more similar circles in the top 16% of the height) stay `rejected` with reason `header_row`.
- An instrument with any unverified connection stays `review`, including the synthetic gapped circle and the real drawing's unverified links. Do not mark them verified to move them onto a layer.
- An OCR box that overlaps a pipe must not delete that pipe. The thick and thin masks are not cleared by the text mask. The synthetic test requires the thick pixel under `DN150` to remain.

## Constraints

Use OpenCV, PaddleOCR, morphology, skeletonization, and the geometric rules already in this package. Dependencies in `requirements.txt`: opencv-python, numpy, scikit-image, paddleocr, paddlepaddle. No YOLO. Do not replace `ossr_p&id` or `P&ID_extraction` graph construction. Change behavior by editing `layer_sep/config.py`; lengths already scale with the measured thin-stroke width.

## Overlay colors

`save_overlay` in `layer_sep/overlay.py` uses OpenCV BGR:

- Thick pipes `(255, 90, 0)` at alpha 0.7; thin pipes `(0, 140, 255)` at 0.7; ambiguous `(180, 180, 180)` at 0.55; text `(0, 220, 255)` at 0.45; instruments `(0, 190, 0)` at 0.35.
- Junction dots `(0, 0, 255)`; visual crossings `(255, 0, 180)`; unresolved crossings `(0, 200, 255)`.
- Free endpoints `(0, 255, 0)`; instrument-port endpoints `(255, 0, 255)`; verified ports are a white ring.

`debug/06_stroke_classes.png` uses the same family with thick `(255, 80, 0)` and instruments `(0, 180, 0)`.
