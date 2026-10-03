"""Stage 3: the wall's true surface, as an offset along W from the picked plane.

The dense point cloud is rasterised onto the plane and each cell keeps its
frontmost point (largest W), so trim, sills and window recesses land at their
real depth instead of being smeared onto a flat plane. That depth is what the
photos get back-projected from, which is what removes parallax at edges.
"""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass
class DepthMap:
    cell_m: float
    height_m: float
    grid: np.ndarray            # (rows, cols) float32 W offset in metres, row 0 = top
    coverage: float             # fraction of cells with real points before fill
    point_count: int
    source: str

    def sample(self, u: np.ndarray, v: np.ndarray) -> np.ndarray:
        if self.grid.size == 1:
            return np.full(np.shape(u), float(self.grid.flat[0]), dtype=np.float64)
        mapx = (np.asarray(u) / self.cell_m - 0.5).astype(np.float32)
        mapy = ((self.height_m - np.asarray(v)) / self.cell_m - 0.5).astype(np.float32)
        out = cv2.remap(self.grid, mapx, mapy, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
        return out.astype(np.float64)

    def stats(self) -> dict:
        g = self.grid
        return {
            "source": self.source,
            "cell_mm": round(self.cell_m * 1000, 2),
            "grid": [int(g.shape[1]), int(g.shape[0])],
            "surface_points": self.point_count,
            "coverage_before_fill": round(self.coverage, 4),
            "offset_min_m": round(float(g.min()), 4),
            "offset_max_m": round(float(g.max()), 4),
            "offset_median_m": round(float(np.median(g)), 4),
        }


def flat_depth(height_m: float, reason: str) -> DepthMap:
    return DepthMap(cell_m=1.0, height_m=height_m, grid=np.zeros((1, 1), np.float32),
                    coverage=0.0, point_count=0, source=f"flat plane ({reason})")


def build_depth_map(wall_pts: np.ndarray, width_m: float, height_m: float, *,
                    cell_m: float, depth_front_m: float, depth_back_m: float,
                    min_coverage: float = 0.02, source: str = "point cloud") -> DepthMap:
    """wall_pts: (N, 3) points already in wall coords (u, v, w)."""
    if wall_pts is None or len(wall_pts) == 0:
        return flat_depth(height_m, "no point cloud")
    u, v, w = wall_pts[:, 0], wall_pts[:, 1], wall_pts[:, 2]
    keep = (u >= 0) & (u < width_m) & (v >= 0) & (v < height_m) & (w <= depth_front_m) & (w >= -depth_back_m)
    u, v, w = u[keep], v[keep], w[keep]
    cols = max(1, int(np.ceil(width_m / cell_m)))
    rows = max(1, int(np.ceil(height_m / cell_m)))
    if len(w) == 0:
        return flat_depth(height_m, "no points near the wall plane")

    ci = np.clip((u / cell_m).astype(np.int64), 0, cols - 1)
    ri = np.clip(((height_m - v) / cell_m).astype(np.int64), 0, rows - 1)
    flat = ri * cols + ci
    grid = np.full(rows * cols, -np.inf, dtype=np.float64)
    np.maximum.at(grid, flat, w)  # frontmost point per cell
    grid = grid.reshape(rows, cols)
    filled = np.isfinite(grid)
    coverage = float(filled.mean())
    if coverage < min_coverage:
        return flat_depth(height_m, f"point coverage {coverage:.1%} is too sparse")

    g32 = np.where(filled, grid, 0.0).astype(np.float32)
    if not filled.all():
        holes = (~filled).astype(np.uint8)
        g32 = cv2.inpaint(g32, holes, 3, cv2.INPAINT_TELEA)
    # Single stray points in front of the wall (noise, a bug, a wire) would
    # otherwise pull a whole cell forward.
    if min(g32.shape) >= 5:
        g32 = cv2.medianBlur(g32, 5)
    elif min(g32.shape) >= 3:
        g32 = cv2.medianBlur(g32, 3)
    return DepthMap(cell_m=cell_m, height_m=height_m, grid=g32, coverage=coverage,
                    point_count=int(len(w)), source=source)
