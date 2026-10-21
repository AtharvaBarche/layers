"""Resolution-aware settings for layer separation.

Pixel lengths scale with the measured thin-stroke width. Nothing here is a
license to label a stroke from width alone.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass
class LayerConfig:
    denoise_ksize: int = 3
    adaptive_block: int = 31
    adaptive_c: int = -2
    min_component_area: int = 3
    deskew: bool = True
    deskew_min_deg: float = 0.8
    deskew_max_deg: float = 12.0
    deskew_min_lines: int = 8

    scale_lengths: bool = True
    min_pipe_length_px: float = 30.0
    min_pipe_lengths_in_widths: float = 8.0
    spur_max_px: float = 10.0
    spur_widths: float = 3.0
    min_linearity: float = 0.70
    pipe_score_threshold: float = 0.56

    width_rel_tol: float = 0.30
    width_abs_tol: float = 1.2
    ambiguous_band_ratio: float = 0.26
    ambiguous_band_px: float = 0.85
    min_peak_separation_ratio: float = 0.40
    min_peak_count: int = 3
    single_mode_thick_min: float = 4.5

    w_width: float = 0.30
    w_length: float = 0.22
    w_linearity: float = 0.28
    w_continuity: float = 0.12
    w_directional: float = 0.08
    direction_kernel_fraction: float = 0.34
    corner_deg: float = 62.0

    instrument_min_confidence: float = 0.58
    instrument_record_confidence: float = 0.40
    min_symbol_side_px: float = 14.0
    min_symbol_side_widths: float = 6.0
    max_symbol_side_fraction: float = 0.28
    circle_circularity: float = 0.70
    grid_min_count: int = 4
    frame_margin_px: float = 6.0
    frame_span_fraction: float = 0.35

    port_endpoint_widths: float = 2.0
    port_endpoint_min_px: float = 7.0
    port_angle_max_deg: float = 28.0
    connection_min_confidence: float = 0.58

    text_max_height_px: float = 48.0
    text_max_height_widths: float = 16.0
    ocr_min_conf: float = 0.45
    text_pipe_max_dist_px: float = 80.0
    text_pipe_max_dist_widths: float = 16.0
    text_assign_min_conf: float = 0.62
    text_assign_margin: float = 0.12

    node_cluster_px: float = 6.0
    crossing_angle_tol: float = 22.0
    junction_blob_ratio: float = 1.7

    rdp_epsilon: float = 1.0
    contour_epsilon: float = 1.0
    cleanup_min_area: int = 12

    hough_threshold: int = 36
    hough_min_length: float = 28.0
    hough_max_gap: float = 8.0
    hough_angle_tol: float = 12.0
    hough_dist_px: float = 4.0

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class ResolvedThresholds:
    """Thresholds after the drawing's own stroke width is known."""

    thin_peak: float | None
    thick_peak: float | None
    scale_width: float
    min_pipe_length: float
    spur_max: float
    port_min_px: float
    text_max_height: float
    text_max_dist: float
    min_symbol_side: float
    max_symbol_side: float
    direction_kernel: int

    def port_reach(self, stroke_width: float) -> float:
        return max(self.port_min_px, stroke_width * 2.0)


def resolve_thresholds(
    cfg: LayerConfig,
    thin_peak: float | None,
    thick_peak: float | None,
    image_shape: tuple[int, int],
) -> ResolvedThresholds:
    scale = thin_peak if thin_peak and thin_peak > 0.4 else 2.0
    min_len = cfg.min_pipe_length_px
    if cfg.scale_lengths:
        min_len = max(min_len, cfg.min_pipe_lengths_in_widths * scale)
    spur = cfg.spur_max_px
    if cfg.scale_lengths:
        spur = max(spur, cfg.spur_widths * scale)
    kernel = int(round(max(9.0, min_len * cfg.direction_kernel_fraction)))
    if kernel % 2 == 0:
        kernel += 1
    short_side = float(min(image_shape))
    return ResolvedThresholds(
        thin_peak=thin_peak,
        thick_peak=thick_peak,
        scale_width=float(scale),
        min_pipe_length=float(min_len),
        spur_max=float(spur),
        port_min_px=float(cfg.port_endpoint_min_px),
        text_max_height=float(max(cfg.text_max_height_px, cfg.text_max_height_widths * scale)),
        text_max_dist=float(max(cfg.text_pipe_max_dist_px, cfg.text_pipe_max_dist_widths * scale)),
        min_symbol_side=float(max(cfg.min_symbol_side_px, cfg.min_symbol_side_widths * scale)),
        max_symbol_side=float(short_side * cfg.max_symbol_side_fraction),
        direction_kernel=kernel,
    )
