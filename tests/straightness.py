"""Measuring wavy architecture: local misregistration of the output vs truth.

A melted sill is detail that is in the right place on average but wanders
up and down along the wall. Phase correlation of small windows against the
ground truth gives the local shift of each window; along a straight edge in
the building every window should agree (and be ~0).
"""
from __future__ import annotations

import cv2
import numpy as np

import synthetic as syn


def local_shifts(rgba, gsd_m, v_lo, v_hi, win_m=0.15, step_m=0.06, search_m=0.09, u_margin_m=0.2):
    """(u centres, dx, dy) in metres: where each small output window really
    came from on the wall, found by matching it against the ground truth
    within +-search_m (the checker repeats every 0.25-0.5 m, so the search is
    kept short of that; phase correlation aliases on it). The 4 cm stripes are
    blurred out first for the same reason: only the checker edges, which
    are 25 cm apart, are matched.

    dy > 0 means the output shows detail higher than it really is."""
    h, w = rgba.shape[:2]
    truth = cv2.cvtColor(syn.truth_image(gsd_m, w, h).astype(np.float32), cv2.COLOR_BGR2GRAY)
    got = cv2.cvtColor(rgba[..., :3][..., ::-1].astype(np.float32), cv2.COLOR_BGR2GRAY)
    sigma = 0.02 / gsd_m
    truth = cv2.GaussianBlur(truth, (0, 0), sigma)
    got = cv2.GaussianBlur(got, (0, 0), sigma)
    alpha = rgba[..., 3] > 0
    win = int(round(win_m / gsd_m))
    s = int(round(search_m / gsd_m))
    r0 = int(round((syn.WALL_H - v_hi) / gsd_m))
    r1 = int(round((syn.WALL_H - v_lo) / gsd_m))
    us, dxs, dys = [], [], []
    for uc in np.arange(u_margin_m, syn.WALL_W - u_margin_m, step_m):
        c0 = int(round(uc / gsd_m)) - win // 2
        if c0 - s < 0 or c0 + win + s > w or r0 - s < 0 or r1 + s > h:
            continue
        if alpha[r0:r1, c0:c0 + win].mean() < 0.95:
            continue
        tpl = got[r0:r1, c0:c0 + win]
        ref = truth[r0 - s:r1 + s, c0 - s:c0 + win + s]
        res = cv2.matchTemplate(ref, tpl, cv2.TM_CCOEFF_NORMED)
        _, _, _, (bx, by) = cv2.minMaxLoc(res)

        def sub(a, i):
            if 0 < i < len(a) - 1:
                den = a[i - 1] - 2 * a[i] + a[i + 1]
                return i + (0.5 * (a[i - 1] - a[i + 1]) / den if abs(den) > 1e-9 else 0.0)
            return float(i)
        fx, fy = sub(res[by], bx), sub(res[:, bx], by)
        us.append(uc)
        dxs.append((fx - s) * gsd_m)
        dys.append(-(fy - s) * gsd_m)
    return np.array(us), np.array(dxs), np.array(dys)


def waviness(rgba, gsd_m, v_lo, v_hi):
    """Spread (std) and worst value of the local vertical shift along the strip, metres."""
    _, _, dy = local_shifts(rgba, gsd_m, v_lo, v_hi)
    return float(np.std(dy)), float(np.max(np.abs(dy - np.median(dy))))


def stripe_offsets(rgba, gsd_m, v_lo, v_hi):
    """Vertical displacement of the 4 cm stripes, per output column, metres.

    The stripes are this wall's sills and mullion lines: dead straight in the
    building. Demodulating them at their known period gives, per column, how
    far up or down the output put them (modulo 4 cm, unwrapped along the
    wall, so smooth bends of any size are followed). A straight facade gives
    a flat line; a melted one wanders."""
    h, w = rgba.shape[:2]
    r0 = int(round((syn.WALL_H - v_hi) / gsd_m))
    r1 = int(round((syn.WALL_H - v_lo) / gsd_m))
    rows = syn.WALL_H - (np.arange(r0, r1) + 0.5) * gsd_m
    carrier = np.exp(-2j * np.pi * rows / 0.04)[:, None]

    def phase(img):
        g = cv2.cvtColor(img.astype(np.float32), cv2.COLOR_BGR2GRAY)
        base = cv2.GaussianBlur(g, (0, 0), 0.02 / gsd_m)
        s = (g / np.maximum(base, 1.0) - 1.0)[r0:r1]
        z = (s * carrier).sum(axis=0)
        z = np.convolve(z, np.ones(5) / 5, mode="same")          # 2.5 cm along the wall
        return z

    truth = syn.truth_image(gsd_m, w, h)
    zt = phase(truth)
    zg = phase(rgba[..., :3][..., ::-1])
    ok = (rgba[r0:r1, :, 3] > 0).all(axis=0) & (np.abs(zg) > 0.2 * np.median(np.abs(zt)))
    dphi = np.angle(zg * np.conj(zt))
    dphi = np.unwrap(np.where(ok, dphi, np.nan)[ok])
    d = np.full(w, np.nan)
    d[ok] = dphi / (2 * np.pi) * 0.04
    return d


def stripe_waviness(rgba, gsd_m, v_lo, v_hi, margin_m=0.15):
    """(std, worst deviation from the median) of the stripe displacement, metres."""
    d = stripe_offsets(rgba, gsd_m, v_lo, v_hi)
    m = int(margin_m / gsd_m)
    d = d[m:-m]
    d = d[np.isfinite(d)]
    return float(np.std(d)), float(np.max(np.abs(d - np.median(d))))
