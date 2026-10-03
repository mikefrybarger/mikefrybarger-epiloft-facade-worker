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
from .selection import score_views


@dataclass
class RefineConfig:
    enabled: bool = True
    cell_m: float = 0.03          # refinement grid
    search_m: float = 0.15        # +- sweep around the cloud depth
    step_m: float = 0.0075
    top_k: int = 4                # photos compared per patch
    window_cells: int = 5         # cost aggregation window (5 x 3 cm = 15 cm)
    min_confidence: float = 0.25  # relative cost drop needed to trust a patch's own depth
    prior_weight: float = 0.35    # pull toward the wall-wide offset (repeating patterns, siding)
    max_local_m: float = 0.08     # a patch may differ from the wall-wide offset by at most this


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
            mx = np.where(inside, (px + 0.5) / f - 0.5, -1).astype(np.float32).reshape(1, -1)
            my = np.where(inside, (py + 0.5) / f - 0.5, -1).astype(np.float32).reshape(1, -1)
            val = cv2.remap(img, mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE).ravel()
            val = np.where(inside, val, 0.0).astype(np.float32)
            wgt = inside.astype(np.float32)
            s1[j][m] += val * wgt
            s2[j][m] += val * val * wgt
            cnt[j][m] += wgt
    del member

    enough = cnt >= 2
    var = np.where(enough, s2 / np.maximum(cnt, 1) - (s1 / np.maximum(cnt, 1)) ** 2, np.nan)
    del s1, s2
    # aggregate over a small window (NaN-aware box filter)
    win = (cfg.window_cells, cfg.window_cells)
    cost = np.empty_like(var)
    for j in range(D):
        vj = var[j]
        ok = np.isfinite(vj).astype(np.float32)
        num = cv2.boxFilter(np.where(np.isfinite(vj), vj, 0).astype(np.float32), -1, win, normalize=False)
        den = cv2.boxFilter(ok, -1, win, normalize=False)
        cost[j] = np.where(den >= 0.6 * win[0] * win[1], num / np.maximum(den, 1e-6), np.nan)
    del var
    full = np.all(np.isfinite(cost), axis=0)
    cmean = np.nanmean(np.where(full[None], cost, np.nan), axis=0)
    norm = cost / np.maximum(cmean, 1e-3)[None]           # 1.0 = no better than average

    def pick(c):
        ci = np.argmin(np.where(np.isfinite(c), c, np.inf), axis=0)
        i0 = np.clip(ci, 1, D - 2)
        cz = np.nan_to_num(c, nan=0.0)
        c_l = np.take_along_axis(cz, (i0 - 1)[None], axis=0)[0]
        c_c = np.take_along_axis(cz, i0[None], axis=0)[0]
        c_r = np.take_along_axis(cz, (i0 + 1)[None], axis=0)[0]
        den = c_l - 2 * c_c + c_r
        frac = np.where(np.abs(den) > 1e-6, 0.5 * (c_l - c_r) / den, 0.0).clip(-0.5, 0.5)
        cmin = np.take_along_axis(np.nan_to_num(c, nan=np.inf), ci[None], axis=0)[0]
        return offsets[i0] + frac * cfg.step_m, ci, cmin

    # pass 1: the wall-wide offset, from clearly textured patches
    best0, ci0, cmin0 = pick(norm)
    conf0 = np.where(full, 1.0 - cmin0, 0.0)
    edge0 = (ci0 == 0) | (ci0 == D - 1)
    strong = full & (conf0 >= cfg.min_confidence) & ~edge0
    if strong.sum() < 20:
        return depth, {"status": "no_texture", "confident_fraction": round(float(strong.mean()), 4)}
    global_off = float(np.median(best0[strong]))
    # pass 2: per patch, with a pull toward the wall-wide offset so a repeating
    # pattern (stripes, siding) cannot lock onto a false match one period away
    pen = cfg.prior_weight * ((offsets - global_off) / cfg.search_m) ** 2
    best, ci, cmin = pick(norm + pen[:, None, None])
    conf = np.where(full, 1.0 - np.take_along_axis(np.nan_to_num(norm, nan=1.0), ci[None], axis=0)[0], 0.0)
    edge = (ci == 0) | (ci == D - 1)
    confident = full & (conf >= cfg.min_confidence) & ~edge
    best = np.clip(best, global_off - cfg.max_local_m, global_off + cfg.max_local_m)
    del cost, norm

    if confident.sum() < 20:
        return depth, {"status": "no_texture", "confident_fraction": round(float(confident.mean()), 4)}
    dw = np.where(confident, best, global_off).astype(np.float32)
    if min(H, W) >= 5:
        dw = cv2.medianBlur(dw, 5)
    new_grid = (base + dw).astype(np.float32)
    lo, hi = float(depth.grid.min()) - cfg.search_m, float(depth.grid.max()) + cfg.search_m
    new_grid = np.clip(new_grid, lo, hi)
    info = {
        "status": "ok",
        "global_offset_m": round(global_off, 4),
        "confident_fraction": round(float(confident.mean()), 4),
        "local_residual_std_m": round(float(np.std(best[confident] - global_off)), 4),
        "image_level": level,
        "photos": n,
    }
    if log:
        log(f"Photo-consistency depth: global offset {global_off * 100:.1f} cm, "
            f"{confident.mean():.0%} of the wall textured enough to refine locally")
    refined = DepthMap(cell_m=cfg.cell_m, height_m=depth.height_m, grid=new_grid,
                       coverage=depth.coverage, point_count=depth.point_count,
                       source=depth.source + " + photo consistency",
                       valid=valid, info={**depth.info, "refine": info})
    return refined, info
