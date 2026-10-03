"""Which face of the picked wall is the outside?

Corners alone define a plane, not a side. Studio knows (the user picked the
wall while looking at it), so the payload can say where the viewer stood
(`wall.view_from`). Without that, the side is decided by evidence: for each
side, how many (photo, wall point) pairs are in a photo's real field of view
and not blocked by the dense geometry. Back-side photos aimed at a building
"point toward" the far wall, but the building blocks it, so they score
nothing. (A plain vote of photos pointing toward the plane chose the back of
the building on the Ascend Plaza job.)
"""
from __future__ import annotations

import numpy as np

from .selection import prefilter_shots
from .visibility import ZBuffer, auto_downscale

SAMPLE_U, SAMPLE_V = 16, 6
MAX_CAMS_PER_SIDE = 48
OCCLUDER_SAMPLE = 1_500_000


def _side_score(shots, plane, normal_sign, occluders, sel, vis_cfg, spacing):
    w = plane.w * normal_sign
    probe = type(plane)(origin=plane.origin, u=plane.u, v=plane.v, w=w,
                        width_m=plane.width_m, height_m=plane.height_m)
    cams = prefilter_shots(shots, probe, sel)
    if not cams:
        return 0, 0
    centre = plane.to_world(plane.width_m / 2, plane.height_m / 2, 0.0)
    cams.sort(key=lambda s: float(np.linalg.norm(s.center - centre)))
    if len(cams) > MAX_CAMS_PER_SIDE:   # spread along the wall, not just the nearest
        idx = np.linspace(0, len(cams) - 1, MAX_CAMS_PER_SIDE).astype(int)
        cams = [cams[i] for i in idx]
    uu, vv = np.meshgrid((np.arange(SAMPLE_U) + 0.5) / SAMPLE_U * plane.width_m,
                         (np.arange(SAMPLE_V) + 0.5) / SAMPLE_V * plane.height_m)
    pts = plane.to_world(uu.ravel(), vv.ravel(), 0.0)
    total = 0
    for shot in cams:
        px, py, d = shot.project(pts)
        width, height = shot.size()
        inside = np.isfinite(px) & (px >= 0) & (py >= 0) & (px <= width - 1) & (py <= height - 1) & (d > 0)
        if not inside.any():
            continue
        dist = float(np.median(d[inside]))
        zb = ZBuffer(shot, occluders, vis_cfg, downscale=vis_cfg.downscale
                     or auto_downscale(shot, dist, spacing, vis_cfg))
        total += int((inside & zb.visible(px, py, d)).sum())
    return total, len(cams)


def choose_side(shots, plane, region, sel, vis_cfg, spacing) -> dict:
    """Orient `plane` by visibility evidence. Returns a report."""
    occ = region
    if occ is not None and len(occ) > OCCLUDER_SAMPLE:
        rng = np.random.default_rng(1)
        occ = occ[rng.choice(len(occ), OCCLUDER_SAMPLE, replace=False)]
    plus, n_plus = _side_score(shots, plane, +1, occ, sel, vis_cfg, spacing)
    minus, n_minus = _side_score(shots, plane, -1, occ, sel, vis_cfg, spacing)
    report = {"method": "visibility", "visible_pairs": {"as_picked": plus, "reverse": minus},
              "photos_tested": {"as_picked": n_plus, "reverse": n_minus}}
    if minus > plus:
        plane.flip("plane turned to the side photos can actually see (visibility check)")
        report["flipped"] = True
    else:
        report["flipped"] = False
    if max(plus, minus) > 0 and min(plus, minus) > 0.5 * max(plus, minus):
        report["warning"] = ("both faces of the wall are visible in many photos; the side was chosen "
                             "by a narrow margin. Send wall.view_from from Studio to make it certain.")
    return report
