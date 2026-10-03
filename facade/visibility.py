"""Stage 4a: occlusion. Is a wall point actually visible in a given photo?

Each candidate photo gets a low-resolution z-buffer built by projecting the
dense point cloud into it (nearest depth per cell, then a small min-filter to
close gaps between points). A wall point is visible when its depth is no more
than a tolerance behind the z-buffer. Trees, posts, porch columns and roof
overhangs are in the point cloud, so they block the wall behind them.

This is point-based visibility rather than mesh ray casting: no Open3D or
Embree dependency, and the dense cloud keeps thin occluders (posts, branches)
that ODM's smoothed mesh tends to lose.
"""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass
class VisibilityConfig:
    downscale: int | None = None  # z-buffer cell size in source pixels; None = from point spacing
    cells_per_spacing: float = 1.5  # auto: cells covered by one point spacing
    close_px: int = 3             # min-filter kernel (z-buffer cells) to fill point gaps
    abs_tol_m: float = 0.08     # always allow this much behind the z-buffer
    rel_tol: float = 0.01       # plus this fraction of the camera distance


def auto_downscale(shot, distance_m: float, point_spacing_m: float, cfg: VisibilityConfig) -> int:
    """z-buffer cell size so neighbouring points land in neighbouring cells.

    Sizing cells in image pixels breaks at high resolution: a 20 MP photo from
    4 m resolves ~1 mm/px, while a dense cloud has points every few cm, so a
    post would be a sieve of single-point cells and the wall would show
    through. Cells are sized so one point spacing covers about
    ``cells_per_spacing`` cells at the wall's distance; the min-filter then
    closes the remaining gaps.
    """
    w, h = shot.size()
    gsd = distance_m / shot.camera.focal_px(w, h)
    return int(np.clip(round(cfg.cells_per_spacing * point_spacing_m / max(gsd, 1e-6)), 2, 128))


class ZBuffer:
    def __init__(self, shot, occluders: np.ndarray, cfg: VisibilityConfig, downscale: int | None = None):
        self.cfg = cfg
        w, h = shot.size()
        s = int(downscale or cfg.downscale or 4)
        self.downscale = s
        self.zw, self.zh = int(np.ceil(w / s)), int(np.ceil(h / s))
        zbuf = np.full(self.zw * self.zh, np.inf, dtype=np.float32)
        self.points_used = 0
        if occluders is not None and len(occluders):
            px, py, d = shot.project(occluders)
            ok = np.isfinite(px) & (d > 0.05) & (px >= -0.5) & (py >= -0.5) & (px < w - 0.5) & (py < h - 0.5)
            if ok.any():
                ix = np.clip(((px[ok] + 0.5) / s).astype(np.int64), 0, self.zw - 1)
                iy = np.clip(((py[ok] + 0.5) / s).astype(np.int64), 0, self.zh - 1)
                np.minimum.at(zbuf, iy * self.zw + ix, d[ok].astype(np.float32))
                self.points_used = int(ok.sum())
        zbuf = zbuf.reshape(self.zh, self.zw)
        if cfg.close_px > 1:
            kernel = np.ones((cfg.close_px, cfg.close_px), np.uint8)
            zbuf = cv2.erode(zbuf, kernel)  # erode == min filter: nearer surfaces win
        self.zbuf = zbuf

    def visible(self, px: np.ndarray, py: np.ndarray, depth: np.ndarray) -> np.ndarray:
        s = self.downscale
        ix = np.clip(((px + 0.5) / s).astype(np.int64), 0, self.zw - 1)
        iy = np.clip(((py + 0.5) / s).astype(np.int64), 0, self.zh - 1)
        z = self.zbuf[iy, ix]
        tol = self.cfg.abs_tol_m + self.cfg.rel_tol * depth
        return depth <= z + tol


def occluder_points(points_wall: np.ndarray, points_world: np.ndarray, width_m: float,
                    height_m: float, depth_back_m: float, reach_m: float, pad_m: float):
    """Points that could stand between the wall and a camera.

    Anything behind the wall surface cannot occlude it. Anything far outside
    the wall's footprint or beyond the farthest camera cannot either.
    """
    if points_wall is None or len(points_wall) == 0:
        return np.zeros((0, 3))
    u, v, w = points_wall[:, 0], points_wall[:, 1], points_wall[:, 2]
    keep = ((w >= -depth_back_m) & (w <= reach_m)
            & (u >= -pad_m) & (u <= width_m + pad_m)
            & (v >= -pad_m) & (v <= height_m + pad_m))
    return points_world[keep]
