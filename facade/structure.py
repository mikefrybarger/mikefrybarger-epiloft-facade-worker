"""Architectural regularisation of the depth model.

Photogrammetry measures a facade's shape well on average and badly at its
edges: a depth step (sill, band over a recessed storefront, sign box edge)
comes out of the cloud smeared and wandering by several cm along the wall.
Wherever the depth model switches surface, the photos are projected from a
different depth, so a wandering switch line bends every sill, header and
mullion it touches (the "melted" storefront bottoms on Ascend Plaza).

Buildings are not like that. So:

* Long boundaries between facade layers are snapped to straight lines
  (horizontal ones nearly horizontal, vertical ones nearly vertical), unless
  the evidence shows a real step in them.
* Every solid structure (sign, column, frame, tower) is one plane, or a few
  planes when it plainly has a few faces, instead of cell-by-cell cloud depth.
"""
from __future__ import annotations

import cv2
import numpy as np

SNAP_BAND_M = 0.10        # how far a measured boundary may wander from its straight line
SNAP_GAP_M = 0.10         # gaps along a boundary bridged when grouping it
SNAP_MIN_LEN_M = 0.5      # boundaries shorter than this are left as they are
SNAP_MAX_SLOPE = 0.03     # straight lines stay within 3 % of horizontal / vertical
STEP_SPLIT_M = 0.06       # a boundary step of this size ...
STEP_PERSIST_M = 1.0      # ... held this long on both sides is real and kept (bigger steps need less)
STRUCT_MAX_LEAN = 0.05    # a structure's plane may lean this much relative to the wall
STRUCT_FACE_SEP_M = 0.06  # depth peaks this far apart are different faces of a structure
STRUCT_FACE_SHARE = 0.15


def _robust_line(x, y, w=None, max_slope=SNAP_MAX_SLOPE, iters=4):
    """y = a + b x by iteratively reweighted least squares, slope capped."""
    w = np.ones_like(x, dtype=np.float64) if w is None else w.astype(np.float64)
    a, b = float(np.median(y)), 0.0
    for _ in range(iters):
        sw = w.sum()
        if sw <= 0 or np.ptp(x) < 1:
            break
        xm, ym = (w * x).sum() / sw, (w * y).sum() / sw
        var = (w * (x - xm) ** 2).sum()
        b = float(np.clip((w * (x - xm) * (y - ym)).sum() / max(var, 1e-9), -max_slope, max_slope))
        a = float(ym - b * xm)
        r = np.abs(y - (a + b * x))
        s = max(1.4826 * np.median(r), 0.5)
        w = w * 0 + 1.0 / np.maximum(1.0, r / (1.5 * s))         # Huber weights
    return a, b


def _split_runs(x, y, min_len, step):
    """Split a boundary into runs at real steps; a wobble stays one run.

    Optimal partitioning of the per-column boundary level into constant
    pieces (least squares + a penalty per extra piece). The penalty is what a
    `step` held for STEP_PERSIST_M on both sides would save, so: a 30 cm jog
    under a window sill splits even over half a metre, a 6 cm step needs about
    a metre on each side, and a +-8 cm wobble that turns around every half
    metre never pays for its pieces."""
    order = np.argsort(x, kind="stable")
    x, y = x[order], y[order]
    ux, inv = np.unique(x, return_inverse=True)
    uy = np.array([np.median(y[inv == i]) for i in range(len(ux))]) if len(ux) < len(x) else y.copy()
    n = len(ux)
    if n < 2 * min_len:
        return [(x, y)]
    persist = STEP_PERSIST_M / STEP_SPLIT_M * step     # cells (step is in cells)
    lam = 0.5 * persist * step * step
    c1 = np.concatenate([[0.0], np.cumsum(uy)])
    c2 = np.concatenate([[0.0], np.cumsum(uy * uy)])
    best = np.zeros(n + 1)
    prev = np.zeros(n + 1, dtype=np.int64)
    for j in range(1, n + 1):
        i = np.arange(j)
        m = j - i
        sse = (c2[j] - c2[i]) - (c1[j] - c1[i]) ** 2 / m
        tot = best[i] + sse + lam
        k = int(np.argmin(tot))
        best[j], prev[j] = tot[k], k
    cuts = []
    j = n
    while j > 0:
        cuts.append((prev[j], j))
        j = prev[j]
    out = []
    for i0, i1 in reversed(cuts):
        sel = (inv >= i0) & (inv < i1)
        out.append((x[sel], y[sel]))
    return out


def _snap_axis(label, cell_m, horizontal: bool, lines: list):
    """Straighten boundaries running along one axis. Returns (label, n_lines).

    Each straight line is appended to `lines` (cell units, in this axis'
    frame: boundary coordinate y = a + b * x, cells x in [x0, x1]; y counts
    cell edges, so the boundary between cell rows r and r + 1 is y = r + 1)."""
    lab = label if horizontal else label.T
    rows, cols = lab.shape
    if rows < 3 or cols < 3:
        return label, 0
    top, bot = lab[:-1], lab[1:]
    edge = (top != bot) & (top >= 0) & (bot >= 0)
    if not edge.any():
        return label, 0
    nl = int(lab.max()) + 1
    pair = np.where(edge, top * nl + bot, -1)
    band = max(1, int(round(SNAP_BAND_M / cell_m)))
    gap = max(1, int(round(SNAP_GAP_M / cell_m)))
    min_len = max(3, int(round(SNAP_MIN_LEN_M / cell_m)))
    step = max(1.5, STEP_SPLIT_M / cell_m)     # a one-cell jog is raster quantisation, not a step
    kernel = np.ones((2 * band + 1, 2 * gap + 1), np.uint8)
    out = lab.copy()
    n_lines = 0
    for p in np.unique(pair[edge]):
        a_lab, b_lab = int(p) // nl, int(p) % nl
        m = (pair == p).astype(np.uint8)
        n, comp = cv2.connectedComponents(cv2.dilate(m, kernel), connectivity=8)
        comp = np.where(m > 0, comp, 0)
        rr, cc = np.nonzero(comp)
        ids = comp[rr, cc]
        for g in range(1, n):
            sel = ids == g
            if not sel.any():
                continue
            gx, gy = cc[sel].astype(np.float64), rr[sel].astype(np.float64) + 1.0   # boundary sits above row r+1
            if np.ptp(gx) + 1 < min_len:
                continue
            for xs, ys in _split_runs(gx, gy, min_len, step):
                if np.ptp(xs) + 1 < min_len:
                    continue
                a, b = _robust_line(xs, ys)
                c0, c1 = int(xs.min()), int(xs.max())
                cs = np.arange(c0, c1 + 1)
                yl = a + b * cs
                r_lo = np.clip(np.floor(yl - band - 1).astype(int), 0, rows - 1)
                r_hi = np.clip(np.ceil(yl + band + 1).astype(int), 0, rows - 1)
                for c, y, lo, hi in zip(cs, yl, r_lo, r_hi):
                    col = out[lo:hi + 1, c]
                    rws = np.arange(lo, hi + 1)
                    ab = (col == a_lab) | (col == b_lab)
                    col[ab] = np.where(rws[ab] + 0.5 < y, a_lab, b_lab)
                lines.append({"axis": "h" if horizontal else "v", "first": a_lab, "second": b_lab,
                              "a": a, "b": b, "x0": c0, "x1": c1, "band": band})
                n_lines += 1
    return (out if horizontal else out.T), n_lines


def straighten_boundaries(label: np.ndarray, cell_m: float):
    """Snap long layer boundaries to straight lines (horizontal, then vertical).

    Returns (label, lines, info)."""
    before = label.copy()
    lines: list = []
    label, nh = _snap_axis(label, cell_m, True, lines)
    label, nv = _snap_axis(label, cell_m, False, lines)
    changed = int((label != before).sum())
    return label, lines, {"straight_lines": {"horizontal": nh, "vertical": nv},
                          "cells_restraightened": changed}


def refine_lines_with_points(lines, pts_u, pts_v, pts_w, planes_coef, nlab, cell_m, height_m):
    """Place each layer-to-layer line where the cloud itself says the switch is.

    The raster puts a boundary on a cell edge (3-9 cm cells); the points that
    built it are denser. Points nearer layer A's plane must sit on A's side,
    points nearer B's on B's; the line is slid to the offset that
    misclassifies the fewest, so a wobbling edge lands on its mean, at
    point-cloud precision. Lines touching a structure keep the raster position.
    Each line gains "shift_m" (along v for horizontal lines, u for vertical)."""
    if not lines or pts_u is None or len(pts_u) == 0:
        return lines
    order = np.argsort(pts_u, kind="stable")
    su, sv, sw = pts_u[order], pts_v[order], pts_w[order]
    for ln in lines:
        ln["shift_m"] = 0.0
        a_lab, b_lab = ln["first"], ln["second"]
        if a_lab >= nlab or b_lab >= nlab:
            continue
        band_m = (ln["band"] + 1) * cell_m
        if ln["axis"] == "h":
            lo, hi = np.searchsorted(su, [ln["x0"] * cell_m, (ln["x1"] + 1) * cell_m])
            u, v, w = su[lo:hi], sv[lo:hi], sw[lo:hi]
            # boundary in metres: v = H - cell * y(x), x continuous column coordinate
            v_line = height_m - cell_m * (ln["a"] + ln["b"] * (u / cell_m - 0.5))
            along = v - v_line              # A (first) is above: positive
        else:
            v_lo, v_hi = height_m - (ln["x1"] + 1) * cell_m, height_m - ln["x0"] * cell_m
            sel = (sv >= v_lo) & (sv <= v_hi)
            u, v, w = su[sel], sv[sel], sw[sel]
            row = (height_m - v) / cell_m - 0.5
            u_line = cell_m * (ln["a"] + ln["b"] * row)
            along = -(u - u_line)           # A (first) is left: make it positive
        near = np.abs(along) < band_m
        if near.sum() < 40:
            continue
        u, v, w, along = u[near], v[near], w[near], along[near]
        pa, pb = planes_coef[a_lab], planes_coef[b_lab]
        da = np.abs(w - (pa[0] + pa[1] * u + pa[2] * v))
        db = np.abs(w - (pb[0] + pb[1] * u + pb[2] * v))
        is_a = (da < db) & (da < 0.07)
        is_b = (db < da) & (db < 0.07)
        if is_a.sum() < 20 or is_b.sum() < 20:
            continue
        # Where is the switch, bin by bin along the line (10 cm bins)? Then a
        # straight line through those local edges: least squares follows
        # their mean, so a wobble cancels out instead of tilting the line.
        pos = u if ln["axis"] == "h" else v
        pivot = float(pos.mean())
        pos = pos - pivot
        ts = np.arange(-band_m, band_m + 1e-9, 0.0025)

        def edge(al, ia, ib):
            sa, sb = np.sort(al[ia]), np.sort(al[ib])
            err = np.searchsorted(sa, ts) + (len(sb) - np.searchsorted(sb, ts, side="right"))
            return float(np.median(ts[err == err.min()]))
        bins = np.floor(pos / 0.10).astype(np.int64)
        bx, by = [], []
        for bi in np.unique(bins):
            m = bins == bi
            if (is_a & m).sum() >= 5 and (is_b & m).sum() >= 5:
                bx.append(float(pos[m].mean()))
                by.append(edge(along[m], is_a[m], is_b[m]))
        if len(bx) >= 4 and np.ptp(bx) > 0.3:
            t, tilt = _robust_line(np.array(bx), np.array(by), max_slope=0.01)
        else:
            t, tilt = edge(along, is_a, is_b), 0.0
        t = float(np.clip(t, -band_m, band_m))
        ln["shift_m"] = t if ln["axis"] == "h" else -t
        ln["tilt"] = tilt if ln["axis"] == "h" else -tilt
        ln["pivot_m"] = pivot
    return lines


def paint_lines_fine(label_f, lines, cell_m, fine_m, height_m):
    """Re-draw the straight lines on the fine raster (in place) at their exact position."""
    rows, cols = label_f.shape
    for ln in lines:
        a_lab, b_lab = ln["first"], ln["second"]
        band_f = int(np.ceil((ln["band"] + 1) * cell_m / fine_m)) + 1
        f = cell_m / fine_m
        if ln["axis"] == "h":
            x0, x1 = int(ln["x0"] * f), min(cols, int(np.ceil((ln["x1"] + 1) * f)))
            xs = np.arange(x0, x1)
            u = (xs + 0.5) * fine_m
            v_line = (height_m - cell_m * (ln["a"] + ln["b"] * (u / cell_m - 0.5)) + ln.get("shift_m", 0.0)
                      + ln.get("tilt", 0.0) * (u - ln.get("pivot_m", 0.0)))
            y = (height_m - v_line) / fine_m                    # fine row-edge coordinate
            target = label_f
        else:
            x0, x1 = int(ln["x0"] * f), min(rows, int(np.ceil((ln["x1"] + 1) * f)))
            xs = np.arange(x0, x1)
            row_c = ((xs + 0.5) * fine_m) / cell_m - 0.5           # coarse row coordinate
            v_c = height_m - (xs + 0.5) * fine_m
            u_line = (cell_m * (ln["a"] + ln["b"] * row_c) + ln.get("shift_m", 0.0)
                      + ln.get("tilt", 0.0) * (v_c - ln.get("pivot_m", 0.0)))
            y = u_line / fine_m
            target = label_f.T
        n_other = target.shape[0]
        for x, yy in zip(xs, y):
            lo, hi = max(0, int(np.floor(yy)) - band_f), min(n_other, int(np.ceil(yy)) + band_f)
            col = target[lo:hi, x]
            idx = np.arange(lo, hi)
            ab = (col == a_lab) | (col == b_lab)
            col[ab] = np.where(idx[ab] + 0.5 < yy, a_lab, b_lab)
    return label_f


def _peaks(vals, sep, share):
    bins = np.arange(vals.min() - 0.02, vals.max() + 0.03, 0.01)
    if len(bins) < 3:
        return [float(np.median(vals))]
    hist, edges = np.histogram(vals, bins=bins)
    smooth = cv2.GaussianBlur(hist.astype(np.float32).reshape(1, -1), (0, 0), 1.5).ravel()
    centres = 0.5 * (edges[:-1] + edges[1:])
    idx = [i for i in range(len(smooth))
           if smooth[i] >= smooth[max(i - 1, 0)] and smooth[i] >= smooth[min(i + 1, len(smooth) - 1)]]
    idx.sort(key=lambda i: -smooth[i])
    chosen = []
    for i in idx:
        c = centres[i]
        if any(abs(c - d) < sep for d in chosen):
            continue
        if np.mean(np.abs(vals - c) < 0.5 * sep) < share:
            continue
        chosen.append(c)
        if len(chosen) == 3:
            break
    return chosen or [float(np.median(vals))]


def _fit_plane(raw, ok, uu, vv):
    """Robust w = a + b u + c v over ok cells, lean capped; constant if too few."""
    w = raw[ok]
    if w.size < 20:
        return np.array([float(np.median(w)) if w.size else 0.0, 0.0, 0.0])
    u, v = uu[ok], vv[ok]
    u0, v0 = u.mean(), v.mean()
    keep = np.ones(w.size, bool)
    coef = np.array([float(np.median(w)), 0.0, 0.0])
    for _ in range(4):
        a = np.stack([np.ones(keep.sum()), u[keep] - u0, v[keep] - v0], -1)
        coef, *_ = np.linalg.lstsq(a, w[keep], rcond=None)
        coef[1:] = np.clip(coef[1:], -STRUCT_MAX_LEAN, STRUCT_MAX_LEAN)
        coef[0] = float(np.median(w[keep] - coef[1] * (u[keep] - u0) - coef[2] * (v[keep] - v0)))
        res = np.abs(w - (coef[0] + coef[1] * (u - u0) + coef[2] * (v - v0)))
        keep = res < max(0.02, 2.5 * np.median(res))
        if keep.sum() < 10:
            break
    return np.array([coef[0] - coef[1] * u0 - coef[2] * v0, coef[1], coef[2]])


def structure_planes(structure: np.ndarray, raw: np.ndarray, reliable: np.ndarray,
                     uc: np.ndarray, vc: np.ndarray, cell_m: float):
    """Depth for structure cells: one plane per face of each structure.

    Returns (face id grid (-1 elsewhere), (n_faces, 3) plane coefficients
    w = a + b u + c v)."""
    faces = np.full(structure.shape, -1, np.int32)
    coefs = []
    n, comp = cv2.connectedComponents(structure.astype(np.uint8), connectivity=8)
    if n <= 1:
        return faces, np.zeros((0, 3))
    boxes = [None] * n
    rr, cc = np.nonzero(comp)
    ids = comp[rr, cc]
    order = np.argsort(ids, kind="stable")
    rr, cc, ids = rr[order], cc[order], ids[order]
    starts = np.searchsorted(ids, np.arange(n))
    ends = np.searchsorted(ids, np.arange(n), side="right")
    for k in range(1, n):
        if ends[k] > starts[k]:
            r, c = rr[starts[k]:ends[k]], cc[starts[k]:ends[k]]
            boxes[k] = (r.min(), r.max() + 1, c.min(), c.max() + 1)
    nf = 0
    win = max(3, int(round(0.09 / cell_m)) | 1)
    for k in range(1, n):
        if boxes[k] is None:
            continue
        r0, r1, c0, c1 = boxes[k]
        m = comp[r0:r1, c0:c1] == k
        rw, ok = raw[r0:r1, c0:c1], reliable[r0:r1, c0:c1] & m
        uu, vv = uc[r0:r1, c0:c1], vc[r0:r1, c0:c1]
        vals = rw[ok]
        peaks = _peaks(vals, STRUCT_FACE_SEP_M, STRUCT_FACE_SHARE) if vals.size >= 20 else [None]
        if len(peaks) == 1:
            coefs.append(_fit_plane(rw, ok, uu, vv))
            faces[r0:r1, c0:c1][m] = nf
            nf += 1
            continue
        # several faces (a sign box on a pylon, a column with a capital): assign
        # each cell to its nearest face depth, tidy, one plane per face
        from .depth import fill_nearest  # noqa: PLC0415 - avoid an import cycle
        filled = fill_nearest(np.where(ok, rw, 0.0), ok) if ok.any() else rw
        lab = np.argmin(np.abs(filled[None] - np.array(peaks)[:, None, None]), axis=0)
        for _ in range(2):
            votes = np.stack([cv2.boxFilter(((lab == j) & m).astype(np.float32), -1, (win, win),
                                            normalize=False) for j in range(len(peaks))])
            lab = np.argmax(votes, axis=0)
        for j in range(len(peaks)):
            sub = m & (lab == j)
            if not sub.any():
                continue
            ns, scomp = cv2.connectedComponents(sub.astype(np.uint8), connectivity=8)
            for s in range(1, ns):
                piece = scomp == s
                pok = piece & ok
                coefs.append(_fit_plane(rw, pok if pok.sum() >= 5 else (ok & sub), uu, vv))
                faces[r0:r1, c0:c1][piece] = nf
                nf += 1
    return faces, np.array(coefs).reshape(-1, 3)
