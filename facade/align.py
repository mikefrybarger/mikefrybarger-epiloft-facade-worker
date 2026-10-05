"""Image-based local alignment: make every photo agree on where detail sits.

Geometry (poses + surface depth) is never perfect, and with oblique captures
(DJI Smart 3D shoots most of a facade from above and from the sides) every
centimetre of depth error moves detail by about a centimetre in the image,
in a direction that depends on each photo's angle. Two photos meeting at a
seam then disagree: doubled house numbers, ghosted lettering.

So, like the local-alignment step of professional orthomosaic tools:

1. Every photo is rendered onto the wall at ~1 cm/px (grey, exposure
   levelled, only where it truly sees the wall).
2. A consensus is formed: the per-pixel median of the photos that see it.
   Photos shot from different angles err in different directions, so the
   median sits near the truth.
3. Dense optical flow (DIS) from each photo to the consensus gives how far
   that photo's detail is off. It is trusted only where there is texture.
4. The flow is split in two. A broad part, smoothed over ~1 m and capped at
   8 cm, absorbs what a photo's pose gets wrong (it moves the whole photo,
   or tilts it, and cannot bend a line). A local part, smoothed over ~0.25 m,
   is only a residual: capped at 3 cm, so it can close a seam but never
   melt a sill. Alignment corrects; the depth model defines the architecture.
5. Two rounds: the consensus is rebuilt from the aligned photos and the flow
   refined.

The result is a smooth warp per photo, applied when the facade is rendered.
Caps are on the length of the shift, not per axis (8 cm per axis allowed
11.3 cm diagonally, which is what v.3 reported on Ascend Plaza).
"""
from __future__ import annotations

import math
import warnings
from dataclasses import dataclass

import cv2
import numpy as np

from .geometry import OrthoGrid
from .images import remap

LUMA = np.array([0.114, 0.587, 0.299])


@dataclass
class AlignConfig:
    enabled: bool = True
    res_m: float = 0.01          # alignment grid
    max_shift_m: float = 0.08    # cap on the broad (per-photo, pose-like) correction
    broad_m: float = 1.0         # the broad correction varies over at least this distance
    max_local_m: float = 0.03    # cap on the local residual on top of it
    smooth_m: float = 0.25       # the local residual varies over at least this distance
    iterations: int = 2
    min_texture: float = 6.0     # gradient magnitude (grey levels/px) to trust flow


class AlignmentField:
    """Per-photo warp: wall (u, v) -> where that photo's detail for (u, v) really is."""

    def __init__(self, grid: OrthoGrid, fields: dict):
        self.grid = grid
        self.fields = fields            # name -> (r0, c0, factor, flow (h, w, 2) float32 in grid px)

    def shift(self, name: str, u: np.ndarray, v: np.ndarray):
        """(du, dv) in metres to add to (u, v) when sampling photo `name`."""
        entry = self.fields.get(name)
        if entry is None:
            return 0.0, 0.0
        r0, c0, f, flow = entry
        g = self.grid
        # continuous index in the full-resolution box, then in the stored (downsampled) field
        x = ((np.asarray(u) / g.gsd_m - c0) / f - 0.5).astype(np.float32)
        y = (((g.height_m - np.asarray(v)) / g.gsd_m - r0) / f - 0.5).astype(np.float32)
        fx = remap(flow[..., 0], x, y, cv2.INTER_LINEAR, border=cv2.BORDER_CONSTANT)
        fy = remap(flow[..., 1], x, y, cv2.INTER_LINEAR, border=cv2.BORDER_CONSTANT)
        return fx * g.gsd_m, -fy * g.gsd_m     # rows run down the wall


def _render(shot, plane, depth, zb, cache, gain_luma, grid, rows, cols, native_gsd_m):
    """Grey ortho of one photo over a window of the alignment grid."""
    r0, r1, c0, c1 = rows[0], rows[1], cols[0], cols[1]
    u, v = grid.uv(r0, r1, c0, c1)
    pts = plane.to_world(u, v, depth.sample(u, v))
    px, py, d = shot.project(pts)
    w, h = shot.size()
    inframe = np.isfinite(px) & (d > 0) & (px >= 0) & (py >= 0) & (px <= w - 1) & (py <= h - 1)
    vis = inframe & zb.visible(px, py, d) & depth.valid_at(u, v)
    level = int(np.clip(math.floor(math.log2(max(grid.gsd_m / 2.0 / max(native_gsd_m, 1e-4), 1.0))), 0, 4))
    img = cache.get(shot, level)
    grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    f = 2.0 ** level
    mx = np.where(vis, (px + 0.5) / f - 0.5, -1).astype(np.float32)
    my = np.where(vis, (py + 0.5) / f - 0.5, -1).astype(np.float32)
    out = remap(grey, mx, my, cv2.INTER_LINEAR).astype(np.float32) * gain_luma
    return np.clip(out, 0, 255).astype(np.uint8), vis


def _cap(flow, cap_px):
    """Limit the length of each shift vector (not each axis)."""
    mag = np.hypot(flow[..., 0], flow[..., 1])
    scale = np.where(mag > cap_px, cap_px / np.maximum(mag, 1e-6), 1.0)
    return (flow * scale[..., None]).astype(np.float32)


def _smooth_flow(flow, weight, sigma_px, cap_px):
    out = np.zeros_like(flow)
    wsum = cv2.GaussianBlur(weight, (0, 0), sigma_px)
    for c in range(2):
        num = cv2.GaussianBlur(flow[..., c] * weight, (0, 0), sigma_px)
        out[..., c] = np.where(wsum > 1e-3, num / np.maximum(wsum, 1e-6), 0.0)
    # fade to zero where there is no evidence at all
    conf = np.clip(wsum / 0.05, 0, 1)
    return _cap(out * conf[..., None], cap_px)


def align_photos(plane, depth, shots, zbuffers, gains, cache, valid_coarse, coarse_cell,
                 native_gsd_m, cfg: AlignConfig, log=None):
    """Returns (AlignmentField or None, report)."""
    if len(shots) < 2:
        return None, {"status": "skipped", "reason": "fewer than two photos"}
    grid = OrthoGrid.for_plane(plane, cfg.res_m)
    H, W = grid.height_px, grid.width_px
    scale = coarse_cell / cfg.res_m
    pad = int(math.ceil((cfg.max_shift_m + cfg.max_local_m) / cfg.res_m)) + 8

    # each photo's footprint on the alignment grid, from the coarse visibility
    boxes = []
    for k in range(len(shots)):
        rr, cc = np.nonzero(valid_coarse[k])
        if rr.size == 0:
            boxes.append(None)
            continue
        r0 = max(0, int(rr.min() * scale) - pad)
        r1 = min(H, int((rr.max() + 1) * scale) + pad)
        c0 = max(0, int(cc.min() * scale) - pad)
        c1 = min(W, int((cc.max() + 1) * scale) + pad)
        boxes.append((r0, r1, c0, c1))

    orthos, masks = {}, {}
    for k, shot in enumerate(shots):
        if boxes[k] is None:
            continue
        r0, r1, c0, c1 = boxes[k]
        g = float(gains[k] @ LUMA)
        orthos[k], masks[k] = _render(shot, plane, depth, zbuffers[shot.name], cache, g, grid,
                                      (r0, r1), (c0, c1), native_gsd_m)

    broad = {k: np.zeros(orthos[k].shape + (2,), np.float32) for k in orthos}
    local = {k: np.zeros(orthos[k].shape + (2,), np.float32) for k in orthos}
    flows = {k: np.zeros(orthos[k].shape + (2,), np.float32) for k in orthos}
    dis = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_MEDIUM)
    sigma = cfg.smooth_m / cfg.res_m
    sigma_b = max(cfg.broad_m, cfg.smooth_m) / cfg.res_m
    cap = cfg.max_shift_m / cfg.res_m
    cap_l = cfg.max_local_m / cfg.res_m
    stats = []
    for it in range(cfg.iterations):
        aligned, amask = {}, {}
        for k in orthos:
            fh, fw = orthos[k].shape
            if it == 0:
                aligned[k], amask[k] = orthos[k], masks[k]
                continue
            yy, xx = np.mgrid[0:fh, 0:fw].astype(np.float32)
            mx, my = xx + flows[k][..., 0], yy + flows[k][..., 1]
            aligned[k] = remap(orthos[k], mx, my, cv2.INTER_LINEAR)
            amask[k] = remap(masks[k].astype(np.uint8), mx, my, cv2.INTER_NEAREST).astype(bool)

        # consensus: per-pixel median of every photo that sees it, in column strips
        consensus = np.zeros((H, W), np.uint8)
        support = np.zeros((H, W), np.uint8)
        strip = 512
        for s0 in range(0, W, strip):
            s1 = min(W, s0 + strip)
            layers = []
            for k in orthos:
                r0, r1, c0, c1 = boxes[k]
                if c1 <= s0 or c0 >= s1:
                    continue
                a0, a1 = max(s0, c0), min(s1, c1)
                buf = np.full((H, s1 - s0), np.nan, np.float32)
                sub = aligned[k][:, a0 - c0:a1 - c0].astype(np.float32)
                subm = amask[k][:, a0 - c0:a1 - c0]
                buf[r0:r1, a0 - s0:a1 - s0] = np.where(subm, sub, np.nan)
                layers.append(buf)
            if not layers:
                continue
            st = np.stack(layers)
            cnt = np.isfinite(st).sum(axis=0)
            with warnings.catch_warnings(), np.errstate(all="ignore"):
                warnings.simplefilter("ignore", RuntimeWarning)     # cells no photo sees
                med = np.nanmedian(st, axis=0)
            consensus[:, s0:s1] = np.nan_to_num(med, nan=0).astype(np.uint8)
            support[:, s0:s1] = np.minimum(cnt, 255).astype(np.uint8)

        gx = cv2.Sobel(consensus, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(consensus, cv2.CV_32F, 0, 1, ksize=3)
        texture = cv2.GaussianBlur(np.hypot(gx, gy) / 4.0, (0, 0), 1.5)
        moved = []
        for k in orthos:
            r0, r1, c0, c1 = boxes[k]
            ref = consensus[r0:r1, c0:c1]
            ok = amask[k] & (support[r0:r1, c0:c1] >= 2)
            if ok.sum() < 500:
                continue
            src = np.where(amask[k], aligned[k], ref)          # outside its view: no motion
            tgt = np.where(ok, ref, src)
            f = dis.calc(tgt, src, None)                          # tgt(p) ~ src(p + f)
            wgt = (ok & (texture[r0:r1, c0:c1] > cfg.min_texture)).astype(np.float32)
            inc_b = _smooth_flow(f, wgt, sigma_b, cap)
            inc_l = _smooth_flow(f - inc_b, wgt, sigma, cap_l)
            broad[k] = _cap(broad[k] + inc_b, cap)
            local[k] = _cap(local[k] + inc_l, cap_l)
            flows[k] = broad[k] + local[k]
            mag = np.hypot(flows[k][..., 0], flows[k][..., 1])[masks[k]]
            moved.append(float(np.percentile(mag, 90)) if mag.size else 0.0)
        stats.append(round(float(np.median(moved)) * cfg.res_m, 4) if moved else 0.0)

    # the warp is smooth over smooth_m, so it is stored at a quarter of the
    # alignment resolution (memory: dozens of photos over a long wall)
    fields = {}
    shifts, broad_p90, local_p90 = [], [], []
    store = max(1, int(round(sigma / 4)))
    for k in orthos:
        r0, r1, c0, c1 = boxes[k]
        m = np.hypot(flows[k][..., 0], flows[k][..., 1])[masks[k]]
        if m.size:
            shifts.append(float(np.percentile(m, 90)) * cfg.res_m)
            broad_p90.append(float(np.percentile(np.hypot(*np.moveaxis(broad[k], -1, 0))[masks[k]], 90))
                             * cfg.res_m)
            local_p90.append(float(np.percentile(np.hypot(*np.moveaxis(local[k], -1, 0))[masks[k]], 90))
                             * cfg.res_m)
        fh, fw = flows[k].shape[:2]
        sh, sw = max(1, -(-fh // store)), max(1, -(-fw // store))
        padded = np.zeros((sh * store, sw * store, 2), np.float32)
        padded[:fh, :fw] = flows[k]
        small = padded.reshape(sh, store, sw, store, 2).mean(axis=(1, 3)).astype(np.float32)
        fields[shots[k].name] = (r0, c0, store, small)
        del flows[k], broad[k], local[k]
    report = {
        "status": "ok",
        "photos": len(fields),
        "p90_shift_m_median": round(float(np.median(shifts)), 4) if shifts else 0.0,
        "p90_shift_m_max": round(float(np.max(shifts)), 4) if shifts else 0.0,
        # broad = pose-like (whole photo); local = bends within a photo. A large
        # broad part means pose error; a large local part means the depth model
        # is still wrong somewhere.
        "broad_p90_m_median": round(float(np.median(broad_p90)), 4) if broad_p90 else 0.0,
        "local_p90_m_median": round(float(np.median(local_p90)), 4) if local_p90 else 0.0,
        "local_p90_m_max": round(float(np.max(local_p90)), 4) if local_p90 else 0.0,
        # photos whose correction is pinned at a cap: alignment was not enough
        "photos_at_cap": int(sum(b >= 0.95 * cfg.max_shift_m or l >= 0.95 * cfg.max_local_m
                                 for b, l in zip(broad_p90, local_p90))),
        "per_iteration_m": stats,
        "grid_mm": round(cfg.res_m * 1000, 1),
    }
    if log:
        log(f"Photo alignment: typical correction {report['p90_shift_m_median'] * 100:.1f} cm "
            f"(broad {report['broad_p90_m_median'] * 100:.1f} cm, local {report['local_p90_m_median'] * 100:.1f} cm), "
            f"largest {report['p90_shift_m_max'] * 100:.1f} cm")
    return AlignmentField(grid, fields), report
