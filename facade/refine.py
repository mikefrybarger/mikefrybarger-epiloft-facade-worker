"""Nudge the wall surface until the photos agree (removes ghosting).

The geometry comes from the dense cloud, and the cloud and the camera poses
never agree perfectly: on the Ascend Plaza job they were ~9 cm apart at the
wall. Seen from photos 5 m away at different angles, a 9 cm depth error
shifts detail by several centimetres between photos, which is exactly the
doubled sign lettering in the output.

So the depth is swept a little forward and back, every good photo is
sampled at each trial depth, and the depth where they agree best (lowest
colour variance) wins. What is allowed to move is structural, not local:

* The whole wall gets one correction line (an offset plus a small drift
  along it): a cloud/pose mismatch is a rigid error, not a per-patch one.
* Each facade layer (glass line, stucco band, sign band) may differ from
  that line by a few cm, as one piece along its whole length. Its fragments
  never get separate corrections: that is what put steps into window
  bottoms where one pane's depth met the next one's.
* A structure face (a sign, a column) may differ from the layer it stands
  on by a few cm more, as one piece.
* Only cells whose depth the cloud really measured vote. Glass shows a
  different reflection in every photo and would pull its region anywhere.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import cv2
import numpy as np

from .depth import DepthMap
from .geometry import OrthoGrid
from .images import remap
from .selection import score_views


@dataclass
class RefineConfig:
    enabled: bool = True
    cell_m: float = 0.03          # refinement grid
    search_m: float = 0.15        # +- sweep around the cloud depth
    step_m: float = 0.0075
    top_k: int = 4                # photos compared per patch
    prior_weight: float = 0.35    # pull toward the wall-wide offset (repeating patterns, siding)
    max_layer_m: float = 0.05     # a facade layer may differ from the wall's correction by at most this
    max_struct_m: float = 0.04    # a structure face may differ from its layer's correction by at most this
    max_drift: float = 0.003      # wall correction may drift this much per metre along the wall
    min_segment_cells: int = 40   # layers / structures smaller than this follow the wall
    min_segment_drop: float = 0.02  # relative cost drop a piece needs to get its own offset


def _gray(img: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32)


def refine_depth(plane, depth: DepthMap, shots, zbuffers, gains, cache, sel, cfg: RefineConfig,
                 native_gsd_m: float, log=None):
    grid = OrthoGrid.for_plane(plane, cfg.cell_m)
    H, W = grid.height_px, grid.width_px
    u, v = grid.uv(0, H, 0, W)
    base = depth.sample(u, v)
    valid = depth.valid_at(u, v)
    n = len(shots)
    if n < 2:
        return depth, {"status": "skipped", "reason": "fewer than two photos"}

    # which photos are best for each patch, at the cloud depth
    pts0 = plane.to_world(u, v, base)
    scores = np.full((n, H, W), -np.inf, dtype=np.float32)
    for i, shot in enumerate(shots):
        px, py, d = shot.project(pts0)
        sc, _ = score_views(shot, pts0, plane.w, px, py, d, sel)
        ok = np.isfinite(sc) & zbuffers[shot.name].visible(px, py, d) & valid
        scores[i] = np.where(ok, sc, -np.inf)
    k = min(cfg.top_k, n)
    top = np.argpartition(-scores, k - 1, axis=0)[:k]                  # (k, H, W)
    top_ok = np.take_along_axis(scores, top, axis=0) > -np.inf
    member = np.zeros((n, H, W), dtype=bool)        # photo i is among the best k for this patch
    for j in range(k):
        for i in np.unique(top[j][top_ok[j]]):
            member[i] |= (top[j] == i) & top_ok[j]
    del scores

    # images at roughly half the refinement cell, grey, exposure-levelled
    level = int(np.clip(math.floor(math.log2(max(cfg.cell_m / 2.0 / max(native_gsd_m, 1e-4), 1.0))), 0, 4))
    luma = np.array([0.114, 0.587, 0.299])
    offsets = np.arange(-cfg.search_m, cfg.search_m + 1e-9, cfg.step_m)
    D = len(offsets)
    s1 = np.zeros((D, H, W), np.float32)
    s2 = np.zeros((D, H, W), np.float32)
    cnt = np.zeros((D, H, W), np.float32)
    for i, shot in enumerate(shots):
        m = member[i]
        if not m.any():
            continue
        img = _gray(cache.get(shot, level)) * float(gains[i] @ luma)
        f = 2.0 ** level
        ih, iw = img.shape
        uu, vv, bb = u[m], v[m], base[m]
        for j, off in enumerate(offsets):
            px, py, d = shot.project(plane.to_world(uu, vv, bb + off))
            inside = np.isfinite(px) & (d > 0) & (px >= 0) & (py >= 0) & (px <= shot.size()[0] - 1) \
                & (py <= shot.size()[1] - 1)
            mx = np.where(inside, (px + 0.5) / f - 0.5, -1).astype(np.float32)
            my = np.where(inside, (py + 0.5) / f - 0.5, -1).astype(np.float32)
            val = remap(img, mx, my, cv2.INTER_LINEAR)
            val = np.where(inside, val, 0.0).astype(np.float32)
            wgt = inside.astype(np.float32)
            s1[j][m] += val * wgt
            s2[j][m] += val * val * wgt
            cnt[j][m] += wgt
    del member

    enough = cnt >= 2
    var = np.where(enough, s2 / np.maximum(cnt, 1) - (s1 / np.maximum(cnt, 1)) ** 2, np.nan)
    del s1, s2, cnt
    full = np.all(np.isfinite(var), axis=0) & valid
    if full.sum() < 50:
        return depth, {"status": "no_overlap", "cells": int(full.sum())}
    # per cell, cost relative to that cell's average over the sweep (1.0 = no preference)
    norm = var / np.maximum(np.nanmean(var, axis=0), 1e-3)[None]
    del var

    # Evidence: measured cells only (glass reflections never vote).
    seg_raw = depth.segment_at(u, v)
    measured = depth.measured_at(u, v)
    use = full & measured
    if use.sum() < 50:
        use = full
    nseg = int(seg_raw.max()) + 1 if (seg_raw.size and seg_raw.max() >= 0) else 1
    seg = np.where(seg_raw >= 0, seg_raw, 0)
    if depth.seg_layer is not None and len(depth.seg_layer) == nseg:
        seg_layer, seg_host = depth.seg_layer, depth.seg_host
    else:                                   # no layer model: the whole wall is one piece
        seg_layer = np.zeros(nseg, dtype=np.int32)
        seg_host = np.zeros(nseg, dtype=np.int32)
    cell_layer = np.where(seg_raw >= 0, seg_host[seg], 0)          # layer each cell stands on
    nlayer = int(seg_host.max()) + 1 if len(seg_host) else 1

    u_mid = float(np.mean(u[use]))
    bin_m = 0.5
    ub = np.clip(((u - u.min()) / bin_m).astype(np.int64), 0, None)
    nb = int(ub.max()) + 1
    ucen = (np.arange(nb) + 0.5) * bin_m + float(u.min()) - u_mid

    def binned(mask):
        """(D, nbins) summed cost and (nbins,) cell counts over mask."""
        b = ub[mask]
        cs = np.stack([np.bincount(b, weights=norm[j][mask], minlength=nb) for j in range(D)])
        return cs, np.bincount(b, minlength=nb).astype(np.float64)

    def line_cost(cs, counts, a_vals, b):
        """Mean cost of the correction line a + b * (u - u_mid) for each a."""
        tot = np.zeros(len(a_vals))
        for i in np.flatnonzero(counts):
            tot += np.interp(a_vals + b * ucen[i], offsets, cs[:, i],
                             left=cs[0, i] * 2, right=cs[-1, i] * 2)
        return tot / max(counts.sum(), 1)

    fine = np.arange(offsets[0], offsets[-1] + 1e-9, cfg.step_m / 4)

    # 1) the wall: offset + drift
    cs_all, n_all = binned(use)
    span = max(float(np.ptp(u[use])), 1e-6)
    max_b = min(cfg.max_drift, cfg.search_m / span)
    best = (np.inf, 0.0, 0.0)
    for b in np.linspace(-max_b, max_b, 13) if max_b > 1e-6 else [0.0]:
        c = line_cost(cs_all, n_all, fine, b)
        i = int(np.argmin(c))
        if c[i] < best[0]:
            best = (float(c[i]), float(fine[i]), float(b))
    wall_cost = line_cost(cs_all, n_all, offsets, 0.0)
    _, global_off, drift = best
    global_drop = float(np.mean(wall_cost) - np.min(wall_cost))

    def wall_line(uu):
        return global_off + drift * (uu - u_mid)

    pen_scale = cfg.prior_weight * 0.1

    def best_delta(cs, counts, base_b, limit, base_off=0.0):
        """Offset (relative to the wall line) minimising cost, or None if not confident."""
        deltas = np.arange(-limit, limit + 1e-9, cfg.step_m / 4)
        c = line_cost(cs, counts, global_off + base_off + deltas, base_b)
        flat = line_cost(cs, counts, offsets, 0.0)
        c = c + pen_scale * ((base_off + deltas) / cfg.search_m) ** 2
        i = int(np.argmin(c))
        drop = float(np.mean(flat) - c[i])
        at_limit = i in (0, len(deltas) - 1)
        if drop < cfg.min_segment_drop * (2.0 if at_limit else 1.0):
            return None
        return float(deltas[i])

    # 2) each layer, as one piece
    layer_delta = np.zeros(nlayer)
    layer_refined = 0
    for k in range(nlayer):
        mk = use & (cell_layer == k) & (np.where(seg_raw >= 0, seg_layer[seg], 0) >= 0)
        if mk.sum() < cfg.min_segment_cells:
            continue
        cs, cn = binned(mk)
        d = best_delta(cs, cn, drift, cfg.max_layer_m)
        if d is not None:
            layer_delta[k] = d
            layer_refined += 1

    # 3) each structure face, as one piece, relative to the layer it stands on
    dseg = layer_delta[seg_host].copy()
    struct_refined = 0
    for k in np.flatnonzero(seg_layer < 0):
        mk = use & (seg_raw == k)
        if mk.sum() < cfg.min_segment_cells:
            continue
        cs, cn = binned(mk)
        d = best_delta(cs, cn, drift, cfg.max_struct_m, base_off=layer_delta[seg_host[k]])
        if d is not None:
            dseg[k] = layer_delta[seg_host[k]] + d
            struct_refined += 1

    # Apply on the depth model's own raster (1 cm), so its straight, sharp
    # steps stay exactly where they are: every piece moves as a whole.
    rows_d, cols_d = depth.grid.shape
    ud = ((np.arange(cols_d) + 0.5) * depth.cell_m)[None, :]
    if depth.segments is not None:
        sd = depth.segments
        piece = np.where(sd >= 0, dseg[np.clip(sd, 0, nseg - 1)], 0.0)
    else:
        piece = 0.0
    new_grid = (depth.grid + wall_line(ud) + piece).astype(np.float32)
    lo, hi = float(depth.grid.min()) - cfg.search_m, float(depth.grid.max()) + cfg.search_m
    new_grid = np.clip(new_grid, lo, hi)
    present = np.bincount(seg[seg_raw >= 0], minlength=nseg) >= cfg.min_segment_cells
    info = {
        "status": "ok",
        "global_offset_m": round(global_off, 4),
        "drift_mm_per_m": round(drift * 1000, 2),
        "global_cost_drop": round(global_drop, 4),
        "segments": nseg,
        "layers_refined": layer_refined,
        "layer_offsets_m": [round(float(x), 4) for x in layer_delta],
        "structures_refined": struct_refined,
        "segments_refined": layer_refined + struct_refined,
        "segment_offset_spread_m": round(float(np.std(dseg[present])) if present.any() else 0.0, 4),
        "evidence_cells": int(use.sum()),
        "glass_cells_ignored": int((full & ~measured).sum()),
        "image_level": level,
        "photos": n,
    }
    if log:
        log(f"Photo-consistency depth: wall offset {global_off * 100:.1f} cm, drift {drift * 1000:.1f} mm/m, "
            f"{layer_refined} layers and {struct_refined} structures corrected as whole pieces")
    refined = DepthMap(cell_m=depth.cell_m, height_m=depth.height_m, grid=new_grid,
                       coverage=depth.coverage, point_count=depth.point_count,
                       source=depth.source + " + photo consistency",
                       valid=depth.valid, info={**depth.info, "refine": info})
    refined.segments = depth.segments
    refined.seg_layer, refined.seg_host = depth.seg_layer, depth.seg_host
    refined.measured = depth.measured
    return refined, info
