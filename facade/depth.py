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

from .images import remap

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
    segments: np.ndarray | None = None  # (rows, cols) int32 region ids (-1 none)

    def segment_at(self, u: np.ndarray, v: np.ndarray) -> np.ndarray:
        if self.segments is None:
            return np.zeros(np.shape(u), dtype=np.int32)
        mx, my, shape = self._maps(u, v)
        out = remap(self.segments.astype(np.float32), mx, my, cv2.INTER_NEAREST)
        return np.rint(out).astype(np.int32).reshape(shape)

    def _maps(self, u, v):
        u, v = np.asarray(u, dtype=np.float64), np.asarray(v, dtype=np.float64)
        shape = u.shape
        mx = (u / self.cell_m - 0.5).astype(np.float32)
        my = ((self.height_m - v) / self.cell_m - 0.5).astype(np.float32)
        return mx, my, shape

    def sample(self, u: np.ndarray, v: np.ndarray) -> np.ndarray:
        if self.grid.size == 1:
            return np.full(np.shape(u), float(self.grid.flat[0]), dtype=np.float64)
        mx, my, shape = self._maps(u, v)
        out = remap(self.grid, mx, my, cv2.INTER_LINEAR)
        return out.astype(np.float64).reshape(shape)

    def valid_at(self, u: np.ndarray, v: np.ndarray) -> np.ndarray:
        if self.valid is None or self.valid.all():
            return np.ones(np.shape(u), dtype=bool)
        mx, my, shape = self._maps(u, v)
        out = remap(self.valid.astype(np.uint8), mx, my, cv2.INTER_NEAREST)
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
                     wall_behind: np.ndarray | None = None, max_wall_behind: float = 0.35,
                     dev: np.ndarray | None = None, obstacle_min_dev: float = 0.45,
                     group_px: int = 0):
    """Connected patches that are big enough and densely measured.

    wall_behind: cells where the cloud also has points on the wall plane. A
    free-standing post or pole has wall behind it (oblique photos see past it);
    a sign, column or tower does not (it is solid). Patches with too much wall
    behind them are obstacles, not facade, and are dropped.

    All patches are scored in one pass (bincount over the label image), so a
    real facade with thousands of small bumps costs no more than one with two.
    """
    m8 = mask.astype(np.uint8)
    if group_px > 0:
        # group nearby pieces (the letters of one sign) before judging size
        k = np.ones((2 * group_px + 1, 2 * group_px + 1), np.uint8)
        m8 = cv2.morphologyEx(m8, cv2.MORPH_CLOSE, k)
    n, lab = cv2.connectedComponents(m8, connectivity=8)
    if n <= 1:
        return np.zeros_like(mask)
    lab = np.where(mask, lab, 0)
    flat = lab.ravel()
    area = np.bincount(flat, minlength=n).astype(np.float64)
    fill = np.bincount(flat, weights=filled.ravel().astype(np.float64), minlength=n) / np.maximum(area, 1)
    ok = (area >= min_cells) & (fill >= min_fill)
    if wall_behind is not None:
        behind = np.bincount(flat, weights=wall_behind.ravel().astype(np.float64), minlength=n) / np.maximum(area, 1)
        far = np.ones(n, dtype=bool)
        if dev is not None:
            # Only things well off the wall can be free-standing. Channel letters
            # and frames sit a few cm out with wall visible between them, but
            # they are mounted on it: they keep their own depth.
            mean_dev = np.bincount(flat, weights=np.abs(dev).ravel().astype(np.float64),
                                   minlength=n) / np.maximum(area, 1)
            far = mean_dev > obstacle_min_dev
        ok &= ~(far & (behind > max_wall_behind))
    ok[0] = False
    return ok[lab] & mask


def sky_mask(filled: np.ndarray, closing_cells: int) -> np.ndarray:
    """Empty space connected to the top edge, after closing small gaps."""
    k = np.ones((closing_cells, closing_cells), np.uint8)
    solid = cv2.morphologyEx(filled.astype(np.uint8), cv2.MORPH_CLOSE, k)
    empty = (solid == 0).astype(np.uint8)
    n, lab = cv2.connectedComponents(empty, connectivity=4)
    top = set(np.unique(lab[0][empty[0] > 0]).tolist()) - {0}
    return np.isin(lab, list(top)) if top else np.zeros_like(filled)


LAYER_TOL_M = 0.07         # a cell within this of a layer plane belongs to it
MAX_LAYERS = 4
MIN_LAYER_SHARE = 0.04     # a layer must cover this fraction of measured cells
MAX_LEAN = 0.005           # largest believable lean or along-wall drift, m per m


def find_layers(raw: np.ndarray, reliable: np.ndarray, cell_m: float, height_m: float):
    """Dominant depth layers of the facade, each fitted as a (slightly leaning) plane.

    A storefront is not one plane: glass lines, stucco bands and sign bands sit
    at different depths. Peaks of the depth histogram are the layers."""
    vals = raw[reliable]
    if vals.size < 50:
        return []
    bins = np.arange(vals.min() - 0.03, vals.max() + 0.04, 0.01)
    hist, edges = np.histogram(vals, bins=bins)
    smooth = cv2.GaussianBlur(hist.astype(np.float32).reshape(1, -1), (0, 0), 2.0).ravel()
    centres = 0.5 * (edges[:-1] + edges[1:])
    peaks = [i for i in range(1, len(smooth) - 1)
             if smooth[i] >= smooth[i - 1] and smooth[i] >= smooth[i + 1]]
    peaks.sort(key=lambda i: -smooth[i])
    chosen = []
    for i in peaks:
        c = centres[i]
        if any(abs(c - d) < 0.10 for d in chosen):
            continue
        if np.mean(np.abs(vals - c) < LAYER_TOL_M) < MIN_LAYER_SHARE:
            continue
        chosen.append(c)
        if len(chosen) == MAX_LAYERS:
            break
    rows, cols = raw.shape
    vv, uu = np.mgrid[0:rows, 0:cols]
    u = (uu + 0.5) * cell_m
    v = height_m - (vv + 0.5) * cell_m
    layers = []
    for c in chosen:
        ok = reliable & (np.abs(raw - c) < LAYER_TOL_M)
        coef = np.array([c, 0.0, 0.0])
        for _ in range(3):
            if ok.sum() < 30:
                break
            a = np.stack([np.ones(ok.sum()), u[ok], v[ok]], -1)
            coef, *_ = np.linalg.lstsq(a, raw[ok], rcond=None)
            res = raw - (coef[0] + coef[1] * u + coef[2] * v)
            ok = reliable & (np.abs(res) < LAYER_TOL_M)
        # Walls are vertical and the picked baseline runs along them, so any
        # real lean is a few mm per metre; more than that is the fit being
        # pulled by soffits, awning undersides or sills. Cap it, then refit
        # the offset alone.
        coef[1:] = np.clip(coef[1:], -MAX_LEAN, MAX_LEAN)
        if ok.sum() >= 30:
            coef[0] = float(np.median(raw[ok] - coef[1] * u[ok] - coef[2] * v[ok]))
        layers.append(coef)
    return layers


def _layer_info(coef, mask, uc, vc):
    """Layer depth at its own centre (an intercept at u = v = 0 can mislead)."""
    if mask.any():
        u0, v0 = float(uc[mask].mean()), float(vc[mask].mean())
    else:
        u0 = v0 = 0.0
    return {"offset_m": round(float(coef[0] + coef[1] * u0 + coef[2] * v0), 4),
            "slope_mm_per_m": [round(float(coef[1]) * 1000, 2), round(float(coef[2]) * 1000, 2)],
            "share": round(float(mask.mean()), 4)}


def build_depth_map(wall_pts: np.ndarray, width_m: float, height_m: float, *,
                    cell_m: float, depth_front_m: float, depth_back_m: float,
                    min_coverage: float = 0.02, source: str = "point cloud",
                    planar_prior: bool = True) -> DepthMap:
    """wall_pts: (N, 3) points already in wall coords (u, v, w).

    Returns a piecewise-planar surface: every cell belongs to one facade layer
    (a plane) or to one solid structure (a sign, column, tower) with its own
    depth, and `segments` numbers the connected regions so later stages can
    correct each region as a whole instead of patch by patch.
    """
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
    flat_idx = ri * cols + ci
    raw, counts = _cell_percentile(flat_idx, w, rows * cols, 0.85)
    raw = raw.reshape(rows, cols)
    counts = counts.reshape(rows, cols)
    filled = np.isfinite(raw)
    coverage = float(filled.mean())
    if coverage < min_coverage:
        return flat_depth(height_m, f"point coverage {coverage:.1%} is too sparse")
    # glass shows the shop interior through it in a few stray points: not a surface
    reliable = filled & (counts >= max(2, 0.25 * np.median(counts[filled])))

    info = {"ground_points_dropped": int(ground.sum())}
    segments = None
    rr, cc = np.mgrid[0:rows, 0:cols]
    uc = (cc + 0.5) * cell_m
    vc = height_m - (rr + 0.5) * cell_m
    layers = find_layers(raw, reliable, cell_m, height_m) if planar_prior else []
    if layers:
        planes = np.stack([c[0] + c[1] * uc + c[2] * vc for c in layers])        # (L, rows, cols)
        res = np.where(reliable[None], raw[None] - planes, np.inf)
        best = np.argmin(np.abs(res), axis=0)
        near = np.take_along_axis(np.abs(res), best[None], axis=0)[0] < LAYER_TOL_M
        label = np.where(reliable & near, best, -1)

        # solid structures standing off (or set into) their nearest layer
        nearest_plane = np.take_along_axis(planes, best[None], axis=0)[0]
        dev = np.where(reliable, raw - nearest_plane, 0.0)
        dev_s = cv2.medianBlur(dev.astype(np.float32), 5) if min(rows, cols) >= 5 else dev
        on_layer = np.zeros(len(w), dtype=bool)
        for c in layers:
            on_layer |= np.abs(w - (c[0] + c[1] * u + c[2] * v)) < 0.05
        wall_behind = (np.bincount(flat_idx[on_layer], minlength=rows * cols).reshape(rows, cols) >= 2)
        cell_area = cell_m * cell_m
        proud = _keep_components((dev_s > LAYER_TOL_M) & reliable, reliable,
                                 int(MIN_STRUCT_AREA_M2 / cell_area), MIN_STRUCT_FILL, wall_behind=wall_behind,
                                 dev=dev_s, group_px=max(1, int(round(0.12 / cell_m))))
        recess = _keep_components((dev_s < -LAYER_TOL_M) & reliable, reliable,
                                  int(MIN_RECESS_AREA_M2 / cell_area), MIN_STRUCT_FILL + 0.15)
        from scipy.ndimage import binary_fill_holes  # noqa: PLC0415
        structure = binary_fill_holes(proud) | binary_fill_holes(recess)

        # everything else takes the layer of its nearest labelled neighbour
        known = label >= 0
        if not known.any():
            known = reliable
            label = np.where(reliable, best, -1)
        label = fill_nearest(label, known)
        nlab = len(layers)
        # tidy layer labels: majority vote so layers form regions, not speckle
        # straight-ish boundaries: facade layer edges run along sills and bands
        win = max(7, int(round(0.22 / cell_m)) | 1)
        for _ in range(3):
            votes = np.stack([cv2.boxFilter((label == k).astype(np.float32), -1, (win, win), normalize=False)
                              for k in range(nlab)])
            label = np.argmax(votes, axis=0)
        grid = np.take_along_axis(planes, label[None], axis=0)[0]
        own = fill_nearest(np.where(reliable, raw, 0.0), reliable)
        if min(rows, cols) >= 5:
            own = cv2.medianBlur(own.astype(np.float32), 5)
        grid = np.where(structure, own, grid)

        # regions: connected pieces of each layer, and each structure
        segments = np.full((rows, cols), -1, dtype=np.int32)
        next_id = 0
        for k in range(nlab):
            n, lab = cv2.connectedComponents(((label == k) & ~structure).astype(np.uint8), connectivity=4)
            segments = np.where(lab > 0, lab - 1 + next_id, segments)
            next_id += n - 1
        n, lab = cv2.connectedComponents(structure.astype(np.uint8), connectivity=8)
        segments = np.where(lab > 0, lab - 1 + next_id, segments).astype(np.int32)
        next_id += n - 1
        info.update({
            "layers": [_layer_info(c, (label == k) & ~structure, uc, vc) for k, c in enumerate(layers)],
            "structure_fraction": round(float(structure.mean()), 4),
            "segments": int(next_id),
            "obstacle_cells_ignored": int(((dev_s > 0.45) & wall_behind & ~proud).sum()),
        })
    else:
        grid = fill_nearest(np.where(filled, raw, 0.0), filled)
        if min(rows, cols) >= 5:
            grid = cv2.medianBlur(grid.astype(np.float32), 5)

    sky = sky_mask(filled, max(3, int(round(0.3 / cell_m))))
    info["sky_fraction"] = round(float(sky.mean()), 4)
    if sky.any():  # trim the ragged fringe where the parapet meets the sky
        k = max(1, int(round(0.04 / cell_m)))
        sky = cv2.dilate(sky.astype(np.uint8), np.ones((2 * k + 1, 2 * k + 1), np.uint8)).astype(bool)
    # The surface can only be inside the search band; never let filtering
    # or filling put it anywhere else.
    grid = np.clip(grid, -depth_back_m, depth_front_m).astype(np.float32)
    dm = DepthMap(cell_m=cell_m, height_m=height_m, grid=grid, coverage=coverage,
                  point_count=int(len(w)), source=source, valid=~sky, info=info)
    dm.segments = segments
    return dm


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
