"""Architectural regularisation of the depth model (v.4).

Direct tests of build_depth_map on point clouds whose real shape is known:
the wavy-sill failure on Ascend Plaza was the depth model switching surface
along the cloud's wandering edge, and signs taking the cloud's noise as shape.
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from facade.depth import build_depth_map  # noqa: E402

W_M, H_M = 8.0, 3.0


def _wall(spacing=0.03, seed=0, noise=0.004):
    rng = np.random.default_rng(seed)
    uu, vv = np.meshgrid(np.arange(0, W_M, spacing), np.arange(0, H_M, spacing))
    u, v = uu.ravel() + rng.uniform(0, spacing, uu.size), vv.ravel() + rng.uniform(0, spacing, vv.size)
    return u, v, rng.normal(0, noise, u.size), rng


def _wobble(u, amp):
    return amp * (0.6 * np.sin(2 * np.pi * u / 0.9) + 0.4 * np.sin(2 * np.pi * u / 0.37 + 1.0))


def _edge_v(dm, threshold):
    """Per fine column: the height where the surface switches across threshold."""
    g = dm.grid
    below = g > threshold                              # lower (proud) part
    first = np.argmax(below, axis=0)                   # first proud row from the top
    return dm.height_m - first * dm.cell_m


def _build(u, v, w, cell=0.06):
    return build_depth_map(np.stack([u, v, w], -1), W_M, H_M, cell_m=cell,
                           depth_front_m=1.0, depth_back_m=1.0)


def test_wobbly_edge_becomes_a_straight_line_at_its_mean():
    u, v, w, _ = _wall()
    sill = 1.9 + _wobble(u, 0.08)                     # the cloud's edge wanders +-8 cm
    w = np.where(v < sill, w + 0.15, w)               # lower part (sill / bulkhead) 15 cm proud
    dm = _build(u, v, w)
    assert dm.cell_m <= 0.011, dm.cell_m              # drawn finely, not on 6 cm cells
    ev = _edge_v(dm, 0.075)[20:-20]
    x = (np.arange(len(ev)) + 20.5) * dm.cell_m
    slope, icpt = np.polyfit(x, ev, 1)
    resid = ev - (icpt + slope * x)
    assert np.max(np.abs(resid)) < 0.0075, np.max(np.abs(resid))   # straight: no bends at all
    assert abs(slope) < 0.004, slope                    # level (the wobble's own trend is -1.3 mm/m)
    assert abs(np.median(ev) - 1.9) < 0.012, np.median(ev)   # and where the edge really is on average
    assert dm.info["straight_lines"]["horizontal"] >= 1, dm.info


def test_a_real_step_in_the_sill_is_kept():
    u, v, w, _ = _wall()
    sill = np.where(u < 4.0, 1.6, 2.1)                # two storefronts, sills 50 cm apart
    w = np.where(v < sill, w + 0.15, w)
    dm = _build(u, v, w)
    ev = _edge_v(dm, 0.075)
    cols = (np.arange(len(ev)) + 0.5) * dm.cell_m
    left, right = ev[(cols > 0.5) & (cols < 3.5)], ev[(cols > 4.5) & (cols < 7.5)]
    assert np.all(np.abs(left - 1.6) < 0.035), (left.min(), left.max())
    assert np.all(np.abs(right - 2.1) < 0.035), (right.min(), right.max())


def test_a_noisy_sign_is_one_flat_plane():
    u, v, w, rng = _wall()
    sign = (u > 2.0) & (u < 3.6) & (v > 2.2) & (v < 2.6)        # 0.64 m2: a structure, not a layer
    # channel letters and a noisy face: +-2 cm of cloud noise, 20 cm proud
    w = np.where(sign, 0.20 + rng.normal(0, 0.02, u.size), w)
    dm = _build(u, v, w)
    g = dm.grid
    uc = (np.arange(g.shape[1]) + 0.5) * dm.cell_m
    vc = dm.height_m - (np.arange(g.shape[0]) + 0.5) * dm.cell_m
    inner = ((uc[None] > 2.15) & (uc[None] < 3.45)) & ((vc[:, None] > 2.3) & (vc[:, None] < 2.5))
    assert abs(float(np.mean(g[inner])) - 0.20) < 0.01, float(np.mean(g[inner]))
    assert float(np.std(g[inner])) < 0.003, float(np.std(g[inner]))    # flat, not the cloud's noise
    # the wall around it is not dragged out by the sign
    around = ((uc[None] > 1.5) & (uc[None] < 4.0)) & ((vc[:, None] > 1.6) & (vc[:, None] < 2.05))
    assert float(np.max(np.abs(g[around]))) < 0.01
    assert dm.info["structure_faces"] >= 1
    faces = dm.seg_layer < 0
    assert faces.sum() >= 1 and np.all(dm.seg_host[faces] >= 0)


def test_glass_takes_its_frame_plane_and_is_not_evidence():
    u, v, w, rng = _wall()
    glass = (u > 1.0) & (u < 3.0) & (v > 0.5) & (v < 2.0)
    keep = ~glass | (rng.random(u.size) < 0.05)       # a few interior points through the glass
    w = np.where(glass, -0.8 + rng.normal(0, 0.3, u.size), w)
    u, v, w = u[keep], v[keep], w[keep]
    dm = _build(u, v, w)
    g = dm.grid
    uc = (np.arange(g.shape[1]) + 0.5) * dm.cell_m
    vc = dm.height_m - (np.arange(g.shape[0]) + 0.5) * dm.cell_m
    pane = ((uc[None] > 1.2) & (uc[None] < 2.8)) & ((vc[:, None] > 0.7) & (vc[:, None] < 1.8))
    assert float(np.max(np.abs(g[pane]))) < 0.01          # one clean plane, the frame's
    assert dm.measured is not None and dm.measured[pane].mean() < 0.2   # and it never votes in refinement
