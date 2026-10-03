"""Stage 3: the wall's true surface, as an offset along W from the picked plane.

What the photos get back-projected from. Getting it wrong by a few
centimetres makes overlapping photos disagree (ghosted letters); getting it
wrong by metres paints pavement onto the wall. A facade is mostly one flat
surface with a few solid things standing off it, so the map is built that way:

1. Rasterise the dense cloud on the plane; per cell take a robust frontmost
   depth (85th percentile, not the single frontmost point, so specks do not
   count).
2. Drop the ground: points low on the wall and in front of it are sidewalk.
3. Fit the wall itself as a plane in (u, v) through cells near the picked
   plane (it may lean or drift a few cm along 50 m).
4. Keep a cell's own depth only where it belongs to a solid structure: a
   connected patch, big enough, densely measured, standing off the wall
   (signs, columns, an entry tower) or set into it (door alcoves). Glass shows
   the interior through it with sparse points, so windows fail that test and
   stay on the wall plane.
5. Empty space connected to the top edge (sky above the parapet) is marked
   invalid, so it comes out transparent instead of stretched.

facade/refine.py then nudges this per patch until the photos agree.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np

GROUND_BAND_M = 0.35      # bottom of the wall where points in front are taken as ground
GROUND_FRONT_M = 0.06
NEAR_PLANE_M = 0.05       # within this of the wall plane = wall
MIN_STRUCT_AREA_M2 = 0.12
MIN_RECESS_AREA_M2 = 0.4
MIN_STRUCT_FILL = 0.55    # fraction of a structure's cells that must hold real points


@dataclass
class DepthMap:
    cell_m: float
    height_m: float
    grid: np.ndarray            # (rows, cols) float32 W offset in metres, row 0 = top
    coverage: float             # fraction of cells with real points before fill
    point_count: int
    source: str
    valid: np.ndarray | None = None     # (rows, cols) bool; False = no surface (sky)
    info: dict = field(default_factory=dict)

    def _maps(self, u, v):
        u, v = np.asarray(u, dtype=np.float64), np.asarray(v, dtype=np.float64)
        shape = u.shape
        mx = (u / self.cell_m - 0.5).astype(np.float32)
        my = ((self.height_m - v) / self.cell_m - 0.5).astype(np.float32)
        if u.ndim != 2:
            mx, my = mx.reshape(1, -1), my.reshape(1, -1)
        return mx, my, shape

    def sample(self, u: np.ndarray, v: np.ndarray) -> np.ndarray:
        if self.grid.size == 1:
            return np.full(np.shape(u), float(self.grid.flat[0]), dtype=np.float64)
        mx, my, shape = self._maps(u, v)
        out = cv2.remap(self.grid, mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
        return out.astype(np.float64).reshape(shape)

    def valid_at(self, u: np.ndarray, v: np.ndarray) -> np.ndarray:
        if self.valid is None or self.valid.all():
            return np.ones(np.shape(u), dtype=bool)
        mx, my, shape = self._maps(u, v)
        out = cv2.remap(self.valid.astype(np.uint8), mx, my, cv2.INTER_NEAREST,
                        borderMode=cv2.BORDER_REPLICATE)
        return out.astype(bool).reshape(shape)

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
            **self.info,
        }


def flat_depth(height_m: float, reason: str) -> DepthMap:
    return DepthMap(cell_m=1.0, height_m=height_m, grid=np.zeros((1, 1), np.float32),
                    coverage=0.0, point_count=0, source=f"flat plane ({reason})")


def _cell_percentile(flat_idx: np.ndarray, w: np.ndarray, n_cells: int, q: float):
    """Per-cell q-quantile of w (NaN where a cell is empty), plus counts."""
    order = np.lexsort((w, flat_idx))
    fi, ws = flat_idx[order], w[order]
    counts = np.bincount(fi, minlength=n_cells)
    starts = np.concatenate([[0], np.cumsum(counts)[:-1]])
    out = np.full(n_cells, np.nan)
    has = counts > 0
    pick = starts[has] + np.floor(q * (counts[has] - 1)).astype(np.int64)
    out[has] = ws[pick]
    return out, counts


def fit_wall_plane(depth: np.ndarray, cell_m: float, height_m: float, iters: int = 4):
    """Robust w = a + b*u + c*v through the dominant surface.

    The dominant surface is the most common depth (histogram peak), so a pick
    that is off the real wall by more than a few cm still finds it."""
    rows, cols = depth.shape
    vv, uu = np.mgrid[0:rows, 0:cols]
    u = (uu + 0.5) * cell_m
    v = height_m - (vv + 0.5) * cell_m
    fin = depth[np.isfinite(depth)]
    if fin.size == 0:
        return np.zeros(3)
    hist, edges = np.histogram(fin, bins=np.arange(fin.min() - 0.02, fin.max() + 0.04, 0.02))
    peak = 0.5 * (edges[np.argmax(hist)] + edges[np.argmax(hist) + 1])
    ok = np.isfinite(depth) & (np.abs(depth - peak) < 0.15)
    coef = np.zeros(3)
    if ok.sum() < 50:
        return coef
    for _ in range(iters):
        a = np.stack([np.ones(ok.sum()), u[ok], v[ok]], -1)
        coef, *_ = np.linalg.lstsq(a, depth[ok], rcond=None)
        resid = depth - (coef[0] + coef[1] * u + coef[2] * v)
        ok = np.isfinite(depth) & (np.abs(resid) < max(0.03, 2.5 * np.nanmedian(np.abs(resid[ok]))))
        if ok.sum() < 50:
            break
    return coef


def _keep_components(mask: np.ndarray, filled: np.ndarray, min_cells: int, min_fill: float,
                     wall_behind: np.ndarray | None = None, max_wall_behind: float = 0.35):
    """Connected patches that are big enough and densely measured.

    wall_behind: cells where the cloud also has points on the wall plane. A
    free-standing post or pole has wall behind it (oblique photos see past it);
    a sign, column or tower does not (it is solid). Patches with too much wall
    behind them are obstacles, not facade, and are dropped."""
    n, lab, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
    keep = np.zeros_like(mask)
    for i in range(1, n):
        comp = lab == i
        area = stats[i, cv2.CC_STAT_AREA]
        if area < min_cells or filled[comp].mean() < min_fill:
            continue
        if wall_behind is not None and wall_behind[comp].mean() > max_wall_behind:
            continue
        keep |= comp
    return keep


def sky_mask(filled: np.ndarray, closing_cells: int) -> np.ndarray:
    """Empty space connected to the top edge, after closing small gaps."""
    k = np.ones((closing_cells, closing_cells), np.uint8)
    solid = cv2.morphologyEx(filled.astype(np.uint8), cv2.MORPH_CLOSE, k)
    empty = (solid == 0).astype(np.uint8)
    n, lab = cv2.connectedComponents(empty, connectivity=4)
    top = set(np.unique(lab[0][empty[0] > 0]).tolist()) - {0}
    return np.isin(lab, list(top)) if top else np.zeros_like(filled)


def build_depth_map(wall_pts: np.ndarray, width_m: float, height_m: float, *,
                    cell_m: float, depth_front_m: float, depth_back_m: float,
                    min_coverage: float = 0.02, source: str = "point cloud",
                    planar_prior: bool = True) -> DepthMap:
    """wall_pts: (N, 3) points already in wall coords (u, v, w)."""
    if wall_pts is None or len(wall_pts) == 0:
        return flat_depth(height_m, "no point cloud")
    u, v, w = wall_pts[:, 0], wall_pts[:, 1], wall_pts[:, 2]
    keep = (u >= 0) & (u < width_m) & (v >= 0) & (v < height_m) & (w <= depth_front_m) & (w >= -depth_back_m)
    ground = (v < GROUND_BAND_M) & (w > GROUND_FRONT_M + 0.5 * v)      # sidewalk at the base
    keep &= ~ground
    u, v, w = u[keep], v[keep], w[keep]
    cols = max(1, int(np.ceil(width_m / cell_m)))
    rows = max(1, int(np.ceil(height_m / cell_m)))
    if len(w) == 0:
        return flat_depth(height_m, "no points near the wall plane")

    ci = np.clip((u / cell_m).astype(np.int64), 0, cols - 1)
    ri = np.clip(((height_m - v) / cell_m).astype(np.int64), 0, rows - 1)
    raw, counts = _cell_percentile(ri * cols + ci, w, rows * cols, 0.85)
    raw = raw.reshape(rows, cols)
    filled = np.isfinite(raw)
    coverage = float(filled.mean())
    if coverage < min_coverage:
        return flat_depth(height_m, f"point coverage {coverage:.1%} is too sparse")

    info = {"ground_points_dropped": int(ground.sum())}
    if planar_prior:
        coef = fit_wall_plane(raw, cell_m, height_m)
        vv, uu = np.mgrid[0:rows, 0:cols]
        plane = (coef[0] + coef[1] * (uu + 0.5) * cell_m
                 + coef[2] * (height_m - (vv + 0.5) * cell_m)).astype(np.float64)
        dev = np.where(filled, raw - plane, 0.0)
        # cells where the cloud also measured the wall itself (behind any object)
        on_plane = np.abs(w - plane[ri, ci]) < NEAR_PLANE_M
        wall_behind = (np.bincount((ri * cols + ci)[on_plane], minlength=rows * cols)
                       .reshape(rows, cols) >= 2)
        # smooth deviations a little so ragged edges do not split structures
        dev_s = cv2.medianBlur(dev.astype(np.float32), 5) if min(rows, cols) >= 5 else dev
        cell_area = cell_m * cell_m
        proud = _keep_components(dev_s > NEAR_PLANE_M, filled, int(MIN_STRUCT_AREA_M2 / cell_area),
                                 MIN_STRUCT_FILL, wall_behind=wall_behind)
        recess = _keep_components(dev_s < -NEAR_PLANE_M, filled, int(MIN_RECESS_AREA_M2 / cell_area),
                                  MIN_STRUCT_FILL + 0.15)
        # windows and gaps inside a solid structure belong to it
        from scipy.ndimage import binary_fill_holes  # noqa: PLC0415
        structure = binary_fill_holes(proud) | binary_fill_holes(recess)
        own = fill_nearest(np.where(filled, raw, 0.0), filled)
        grid = np.where(structure, own, plane)
        if min(rows, cols) >= 5:
            grid = cv2.medianBlur(grid.astype(np.float32), 5)
        info.update({
            "wall_plane_offset_m": round(float(coef[0]), 4),
            "wall_plane_slope_mm_per_m": [round(float(coef[1]) * 1000, 3), round(float(coef[2]) * 1000, 3)],
            "structure_fraction": round(float(structure.mean()), 4),
            "obstacle_cells_ignored": int(((dev_s > NEAR_PLANE_M) & wall_behind & ~proud).sum()),
        })
    else:
        grid = fill_nearest(np.where(filled, raw, 0.0), filled)
        if min(rows, cols) >= 5:
            grid = cv2.medianBlur(grid.astype(np.float32), 5)

    sky = sky_mask(filled, max(3, int(round(0.3 / cell_m))))
    info["sky_fraction"] = round(float(sky.mean()), 4)
    # The surface can only be inside the search band; never let filtering
    # or filling put it anywhere else.
    grid = np.clip(grid, -depth_back_m, depth_front_m).astype(np.float32)
    return DepthMap(cell_m=cell_m, height_m=height_m, grid=grid, coverage=coverage,
                    point_count=int(len(w)), source=source, valid=~sky, info=info)


def fill_nearest(grid: np.ndarray, known: np.ndarray) -> np.ndarray:
    """Fill unknown cells with the value of the nearest known cell.

    Deliberately simple: the result can never leave the range of the real
    measurements. (cv2.inpaint was used here first; on float data it
    extrapolated -0.3..0.2 m depths to -1.8..1.5 m, which threw every
    back-projected pixel metres off the wall on a real job.)
    """
    if known.all():
        return grid
    from scipy.ndimage import distance_transform_edt  # noqa: PLC0415

    _, (iy, ix) = distance_transform_edt(~known, return_indices=True)
    return grid[iy, ix]
