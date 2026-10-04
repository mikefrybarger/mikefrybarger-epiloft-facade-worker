"""Nudge the wall surface until the photos agree (removes ghosting).

The geometry comes from the dense cloud, and the cloud and the camera poses
never agree perfectly: on the Ascend Plaza job they were ~9 cm apart at the
wall. Seen from photos 5 m away at different angles, a 9 cm depth error
shifts detail by several centimetres between photos, which is exactly the
doubled sign lettering in the output.

So for each patch of wall, the depth is swept a little forward and back,
every good photo of that patch is sampled at each trial depth, and the depth
where they agree best (lowest colour variance, aggregated over a small
window) wins. Confident patches (texture, lettering, frames) get their own
correction; flat paint, where any depth looks the same, takes the wall-wide
median correction, so a global cloud/pose offset is removed everywhere.
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
    max_local_m: float = 0.08     # a region may differ from the wall-wide offset by at most this
    min_segment_cells: int = 40   # regions smaller than this take the wall-wide offset
    min_segment_drop: float = 0.02  # relative cost drop a region needs to get its own offset


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

    # One correction per region (a facade layer piece, a sign, a column), from
    # every cell in it. Patch-by-patch corrections jitter on glass, whose
    # reflections differ in every photo; a whole region cannot wobble.
    seg = depth.segment_at(u, v)
    seg = np.where(seg >= 0, seg, seg.max() + 1 if seg.size else 0)
    nseg = int(seg.max()) + 1
    sf = seg[full]
    sums = np.stack([np.bincount(sf, weights=norm[j][full], minlength=nseg) for j in range(D)])  # (D, S)
    ncell = np.bincount(sf, minlength=nseg).astype(np.float64)
    seg_cost = sums / np.maximum(ncell, 1)[None]
    wall_cost = sums.sum(axis=1) / max(ncell.sum(), 1)

    def parabola(c, i):
        i0 = int(np.clip(i, 1, D - 2))
        den = c[i0 - 1] - 2 * c[i0] + c[i0 + 1]
        frac = 0.5 * (c[i0 - 1] - c[i0 + 1]) / den if abs(den) > 1e-9 else 0.0
        return float(offsets[i0] + np.clip(frac, -0.5, 0.5) * cfg.step_m)

    gi = int(np.argmin(wall_cost))
    global_off = parabola(wall_cost, gi)
    global_drop = float(np.mean(wall_cost) - wall_cost[gi])
    pen = cfg.prior_weight * 0.1 * ((offsets - global_off) / cfg.search_m) ** 2
    dw_seg = np.full(nseg, global_off)
    refined_segments = 0
    for k in range(nseg):
        if ncell[k] < cfg.min_segment_cells:
            continue
        c = seg_cost[:, k] + pen
        i = int(np.argmin(c))
        drop = float(np.mean(seg_cost[:, k]) - seg_cost[i, k])
        if i in (0, D - 1) or drop < cfg.min_segment_drop:
            continue
        dw_seg[k] = np.clip(parabola(c, i), global_off - cfg.max_local_m, global_off + cfg.max_local_m)
        refined_segments += 1
    dw = dw_seg[seg].astype(np.float32)
    new_grid = (base + dw).astype(np.float32)
    lo, hi = float(depth.grid.min()) - cfg.search_m, float(depth.grid.max()) + cfg.search_m
    new_grid = np.clip(new_grid, lo, hi)
    big = ncell >= cfg.min_segment_cells
    info = {
        "status": "ok",
        "global_offset_m": round(global_off, 4),
        "global_cost_drop": round(global_drop, 4),
        "segments": nseg,
        "segments_refined": refined_segments,
        "segment_offset_spread_m": round(float(np.std(dw_seg[big] - global_off)) if big.any() else 0.0, 4),
        "image_level": level,
        "photos": n,
    }
    if log:
        log(f"Photo-consistency depth: wall offset {global_off * 100:.1f} cm, "
            f"{refined_segments} of {nseg} regions corrected individually")
    refined = DepthMap(cell_m=cfg.cell_m, height_m=depth.height_m, grid=new_grid,
                       coverage=depth.coverage, point_count=depth.point_count,
                       source=depth.source + " + photo consistency",
                       valid=valid, info={**depth.info, "refine": info})
    return refined, info
