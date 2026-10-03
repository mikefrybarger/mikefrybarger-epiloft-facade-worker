"""Stage 4b and 5a: which photo paints each part of the wall, and exposure gains.

A coarse pass over the wall (a few hundred cells on the long edge) scores
every candidate photo at every cell:

    score = cos(incidence)^2 * centre_weight / native_gsd

so a photo wins when it looks at the wall square-on, from close up, with the
spot near the middle of the frame (least lens distortion and vignetting). The
per-cell winner is then smoothed so seams follow large regions instead of
speckling, and the same pass collects the colour overlaps used to solve
per-photo exposure gains.
"""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass
class SelectionConfig:
    max_incidence_deg: float = 65.0   # reject views more oblique than this
    min_standoff_m: float = 0.5       # camera must be at least this far in front of the wall
    edge_margin: float = 0.02         # ignore this fraction of the frame at each edge
    max_cameras: int = 80             # cap on photos used per wall
    top_per_cell: int = 6             # photos occlusion-checked per wall spot, per round
    min_cells: int = 4                # photo must win or cover this many coarse cells
    smooth_passes: int = 2            # mode-filter passes on the coarse label map
    gain_sigma_n: float = 10.0        # intensity noise, 0-255 scale (OpenCV's default)
    gain_sigma_g: float = 1.0         # weak pull toward gain 1: only anchors overall brightness.
                                      # OpenCV's 0.1 leaves ~40% of a real exposure step uncorrected.
    gain_clamp: tuple = (0.5, 2.0)


def score_views(shot, points, normal, px, py, depth, cfg: SelectionConfig):
    """Score (..., ) for a photo viewing world points; -inf where unusable."""
    w, h = shot.size()
    to_cam = shot.center - points
    dist = np.linalg.norm(to_cam, axis=-1)
    with np.errstate(invalid="ignore", divide="ignore"):
        cos_inc = (to_cam @ normal) / dist
    mx, my = cfg.edge_margin * w, cfg.edge_margin * h
    inside = (np.isfinite(px) & (px >= mx) & (py >= my) & (px <= w - 1 - mx) & (py <= h - 1 - my)
              & (depth > 0.05))
    ok = inside & (cos_inc >= np.cos(np.radians(cfg.max_incidence_deg)))
    rx = (px - (w - 1) / 2.0) / (w / 2.0)
    ry = (py - (h - 1) / 2.0) / (h / 2.0)
    centre = 1.0 - 0.3 * np.clip((rx * rx + ry * ry) / 2.0, 0.0, 1.0)
    gsd = shot.native_gsd(np.maximum(depth, 1e-3))
    score = (cos_inc ** 2) * centre / gsd
    return np.where(ok, score, -np.inf), gsd


def prefilter_shots(shots, plane, cfg: SelectionConfig):
    """Cheap geometric gate before any projection work."""
    keep = []
    cos_max = np.cos(np.radians(cfg.max_incidence_deg + 15.0))  # axis may be off the wall centre
    for s in shots:
        c = s.center
        standoff = float((c - plane.origin) @ plane.w)
        if standoff < cfg.min_standoff_m:
            continue
        if float(s.optical_axis @ -plane.w) < cos_max:
            continue
        keep.append(s)
    return keep


def mode_filter(labels: np.ndarray, valid: np.ndarray, passes: int) -> np.ndarray:
    """Majority vote in a 5x5 window, restricted to photos valid at each cell."""
    n = valid.shape[0]
    out = labels.copy()
    for _ in range(passes):
        votes = np.empty(valid.shape, np.float32)
        for k in range(n):
            votes[k] = cv2.boxFilter((out == k).astype(np.float32), -1, (5, 5), normalize=False,
                                     borderType=cv2.BORDER_REPLICATE)
        votes[~valid] = -1.0
        best = votes.argmax(axis=0)
        out = np.where(valid.any(axis=0), best, -1)
    return out


def solve_gains(samples: np.ndarray, valid: np.ndarray, cfg: SelectionConfig) -> np.ndarray:
    """Per-photo, per-channel gains from overlapping coarse samples.

    samples: (n, cells, 3) float colour at each coarse cell; valid: (n, cells).
    Least squares on OpenCV's GainCompensator objective:
        sum_ij N_ij * [ (g_i I_ij - g_j I_ji)^2 / sN^2 ]  +  sum_i N_i (1 - g_i)^2 / sg^2
    """
    n = samples.shape[0]
    gains = np.ones((n, 3), dtype=np.float64)
    if n < 2:
        return gains
    alpha, beta = 1.0 / cfg.gain_sigma_n ** 2, 1.0 / cfg.gain_sigma_g ** 2
    for c in range(3):
        rows, rhs = [], []
        totals = np.zeros(n)
        for i in range(n):
            for j in range(i + 1, n):
                both = valid[i] & valid[j]
                nij = int(both.sum())
                if nij < 3:
                    continue
                iij = float(samples[i, both, c].mean())
                iji = float(samples[j, both, c].mean())
                row = np.zeros(n)
                wgt = np.sqrt(nij * alpha)
                row[i], row[j] = wgt * iij, -wgt * iji
                rows.append(row)
                rhs.append(0.0)
                totals[i] += nij
                totals[j] += nij
        for i in range(n):
            if totals[i] == 0:
                continue
            row = np.zeros(n)
            wgt = np.sqrt(totals[i] * beta)
            row[i] = wgt
            rows.append(row)
            rhs.append(wgt)
        if not rows:
            continue
        sol, *_ = np.linalg.lstsq(np.asarray(rows), np.asarray(rhs), rcond=None)
        sol = np.where(totals > 0, sol, 1.0)
        gains[:, c] = sol
    # The pairwise terms only fix gains relative to each other. With many
    # photos and imperfect overlaps, shrinking every gain lowers that cost,
    # so the raw solution drifts dark (a real 80-photo job sat every photo on
    # the 0.5 floor). Rescale so the facade keeps the capture's typical
    # exposure: an overlap-weighted, trimmed geometric mean of the gains is 1.
    weight = valid.sum(axis=1).astype(np.float64)
    if weight.sum() > 0:
        lum = np.maximum(gains @ np.array([0.114, 0.587, 0.299]), 1e-6)   # BGR luma
        logs = np.log(lum)
        lo, hi = np.percentile(logs, [10, 90]) if n >= 10 else (logs.min(), logs.max())
        keep = (logs >= lo) & (logs <= hi) & (weight > 0)
        if keep.any():
            gains /= np.exp(np.average(logs[keep], weights=weight[keep]))
    return np.clip(gains, *cfg.gain_clamp)
