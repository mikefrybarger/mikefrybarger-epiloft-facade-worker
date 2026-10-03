"""Work out which frame the camera poses are in, from the data, not file names.

ODM writes ``reconstruction.json`` in one of two frames:

* **offset** (georeferenced runs): UTM minus the ``coords.txt`` offset. ODM
  keeps the original as ``reconstruction.topocentric.json``.
* **topocentric**: OpenSfM's local east-north-up frame around
  ``reference_lla.json``.

Archives do not always carry the marker file (WebODM Lightning's ``all.zip``
can omit it), and treating topocentric poses as offset poses is not a small
error: grid convergence alone rotates the site by about a degree in western
South Dakota, tens of centimetres across one wall. So both hypotheses are
tested against georeferenced geometry whose frame is never in doubt (the
LAZ, absolute UTM, or ODM's textured mesh, which is in the offset frame and
is what Studio displays), and the one that fits is kept. A final translation-only snap
removes any leftover datum offset, so the photos line up with the same
geometry Studio's mesh came from.
"""
from __future__ import annotations

import numpy as np

GOOD_FIT_M = 0.30      # median sparse-to-dense distance that counts as aligned
MAX_SNAP_M = 3.0       # never "fix" a residual larger than this by translation alone
SPARSE_SAMPLE = 20_000
TREE_SAMPLE = 2_000_000


def umeyama(src: np.ndarray, dst: np.ndarray):
    """Similarity (s, R, t) minimising |s R src + t - dst|."""
    mu_s, mu_d = src.mean(0), dst.mean(0)
    a, b = src - mu_s, dst - mu_d
    cov = b.T @ a / len(src)
    u, d, vt = np.linalg.svd(cov)
    sign = np.eye(3)
    if np.linalg.det(u) * np.linalg.det(vt) < 0:
        sign[2, 2] = -1
    r = u @ sign @ vt
    var = (a ** 2).sum() / len(src)
    s = float(np.trace(np.diag(d) @ sign) / var)
    return s, r, mu_d - s * r @ mu_s


def topocentric_to_offset(ref_lla: dict, epsg: int, offset_e: float, offset_n: float, extent_pts: np.ndarray):
    """Local similarity taking OpenSfM topocentric coords to the offset frame.

    Exact over a site: the true map (ENU -> geodetic -> projected) is fitted
    by a similarity on a grid spanning the site, and the fit residual is
    returned so it can be checked.
    """
    from pyproj import Transformer  # noqa: PLC0415

    lo, hi = extent_pts.min(0) - 10.0, extent_pts.max(0) + 10.0
    g = np.stack(np.meshgrid(*[np.linspace(lo[i], hi[i], 5) for i in range(3)]), -1).reshape(-1, 3)
    to_geo = Transformer.from_pipeline(
        "+proj=pipeline "
        f"+step +inv +proj=topocentric +ellps=WGS84 +lat_0={ref_lla['lat']} "
        f"+lon_0={ref_lla['lon']} +h_0={ref_lla['alt']} "
        "+step +inv +proj=cart +ellps=WGS84 "
        "+step +proj=unitconvert +xy_in=rad +xy_out=deg"
    )
    lon, lat, h = to_geo.transform(g[:, 0], g[:, 1], g[:, 2])
    to_proj = Transformer.from_crs("EPSG:4326", f"EPSG:{epsg}", always_xy=True)
    e, n = to_proj.transform(lon, lat)
    dst = np.stack([np.asarray(e) - offset_e, np.asarray(n) - offset_n, np.asarray(h)], -1)
    s, r, t = umeyama(g, dst)
    resid = float(np.abs((s * g @ r.T + t) - dst).max())
    return s, r, t, resid


def apply_similarity(shots: dict, s: float, r: np.ndarray, t: np.ndarray):
    """Move every shot by X' = s R X + t (orientation follows R)."""
    for shot in shots.values():
        c = s * r @ shot.center + t
        shot.R = shot.R @ r.T
        shot.t = -shot.R @ c


def apply_translation(shots: dict, delta: np.ndarray):
    for shot in shots.values():
        c = shot.center + delta
        shot.t = -shot.R @ c


def _fit(points: np.ndarray, tree):
    d, idx = tree.query(points, k=1, workers=-1)
    return float(np.median(d)), idx


def resolve_frame(shots: dict, sparse: np.ndarray | None, dense_offset: np.ndarray | None,
                  ref_lla: dict | None, epsg: int | None, offset_e: float, offset_n: float,
                  marker_present: bool):
    """Decide the pose frame and move shots into the offset frame.

    Returns (report, transform) where transform(points) applies the same move
    to anything else stored in the reconstruction's original frame (the PLY).
    """
    identity = lambda p: p  # noqa: E731
    report = {"method": None, "hypotheses": {}, "snap_m": [0.0, 0.0, 0.0], "fit_m": None, "warnings": []}
    can_topo = ref_lla is not None and epsg is not None
    have_check = sparse is not None and len(sparse) >= 50 and dense_offset is not None and len(dense_offset) >= 1000

    if not have_check:
        if marker_present or not can_topo:
            report["method"] = "offset (marker file)" if marker_present else "offset (assumed: no reference_lla)"
        else:
            report["method"] = "offset (assumed: no georeferenced point cloud to check against)"
            report["warnings"].append(
                "could not verify the camera frame (no georeferenced LAZ or textured mesh in "
                "the archive); if the facade looks shifted or rotated, include "
                "odm_georeferencing or odm_texturing in the ZIP"
            )
        return report, identity

    from scipy.spatial import cKDTree  # noqa: PLC0415

    rng = np.random.default_rng(0)
    if len(dense_offset) > TREE_SAMPLE:
        dense_offset = dense_offset[rng.choice(len(dense_offset), TREE_SAMPLE, replace=False)]
    if len(sparse) > SPARSE_SAMPLE:
        sparse = sparse[rng.choice(len(sparse), SPARSE_SAMPLE, replace=False)]
    tree = cKDTree(dense_offset)

    fit_offset, _ = _fit(sparse, tree)
    report["hypotheses"]["offset"] = round(fit_offset, 4)
    choice, sim = "offset", None
    if can_topo:
        s, r, t, resid = topocentric_to_offset(ref_lla, epsg, offset_e, offset_n, sparse)
        fit_topo, _ = _fit(s * sparse @ r.T + t, tree)
        report["hypotheses"]["topocentric"] = round(fit_topo, 4)
        report["topocentric_rotation_deg"] = round(float(np.degrees(np.arctan2(r[1, 0], r[0, 0]))), 4)
        report["topocentric_fit_residual_m"] = round(resid, 6)
        if fit_topo < fit_offset:
            choice, sim = "topocentric", (s, r, t)

    if sim is not None:
        apply_similarity(shots, *sim)
        sparse = sim[0] * sparse @ sim[1].T + sim[2]
    report["method"] = f"{choice} (point cloud check)"

    # translation-only snap onto the dense cloud
    snap = np.zeros(3)
    pts = sparse.copy()
    best, idx = _fit(pts, tree)
    for _ in range(4):
        d, idx = tree.query(pts, k=1, workers=-1)
        near = d < max(3 * np.median(d), 0.5)
        if near.sum() < 50:
            break
        delta = np.median(dense_offset[idx[near]] - pts[near], axis=0)
        trial = pts + delta
        fit, _ = _fit(trial, tree)
        if fit >= best - 1e-4:
            break
        pts, best, snap = trial, fit, snap + delta
    if np.linalg.norm(snap) > MAX_SNAP_M:
        report["warnings"].append(
            f"poses and point cloud differ by {np.linalg.norm(snap):.2f} m; not snapping, check the dataset"
        )
        snap = np.zeros(3)
        best, _ = _fit(sparse, tree)
    elif np.linalg.norm(snap) > 0:
        apply_translation(shots, snap)
    report["snap_m"] = [round(float(v), 4) for v in snap]
    report["fit_m"] = round(best, 4)
    if best > GOOD_FIT_M:
        report["warnings"].append(
            f"camera poses only fit the point cloud to {best:.2f} m (median); the facade may be misaligned"
        )

    def transform(p):
        p = np.asarray(p, dtype=np.float64)
        if sim is not None:
            p = sim[0] * p @ sim[1].T + sim[2]
        return p + snap
    return report, transform
