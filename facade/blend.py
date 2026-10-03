"""Stage 5b: multiband (Laplacian pyramid) blending of the per-pixel photo choice.

Low frequencies (overall brightness) blend across a wide band either side of
a seam, high frequencies (siding lines, brick edges) switch over within a few
pixels, so seams disappear without ghosting fine detail.
"""
from __future__ import annotations

import cv2
import numpy as np


def _gaussian_pyr(img, levels):
    pyr = [img]
    for _ in range(levels):
        pyr.append(cv2.pyrDown(pyr[-1]))
    return pyr


def _laplacian_pyr(img, levels):
    gp = _gaussian_pyr(img, levels)
    lp = []
    for i in range(levels):
        up = cv2.pyrUp(gp[i + 1], dstsize=(gp[i].shape[1], gp[i].shape[0]))
        lp.append(gp[i] - up)
    lp.append(gp[-1])
    return lp


def max_levels(h: int, w: int, wanted: int) -> int:
    lv = 0
    while lv < wanted and min(h, w) >> (lv + 1) >= 4:
        lv += 1
    return lv


def multiband_blend(images, masks, levels: int = 5) -> np.ndarray:
    """images: list of (H, W, 3) float32, already filled everywhere.
    masks: list of (H, W) float32 in [0, 1], summing to 1 where covered.
    """
    if len(images) == 1:
        return images[0]
    h, w = masks[0].shape
    levels = max_levels(h, w, levels)
    acc = None
    wsum = None
    for img, m in zip(images, masks):
        lp = _laplacian_pyr(img, levels)
        gm = _gaussian_pyr(m, levels)
        if acc is None:
            acc = [np.zeros_like(x) for x in lp]
            wsum = [np.zeros_like(x) for x in gm]
        for i in range(levels + 1):
            acc[i] += lp[i] * gm[i][..., None]
            wsum[i] += gm[i]
    for i in range(levels + 1):
        acc[i] /= np.maximum(wsum[i], 1e-6)[..., None]
    out = acc[-1]
    for i in range(levels - 1, -1, -1):
        out = cv2.pyrUp(out, dstsize=(acc[i].shape[1], acc[i].shape[0])) + acc[i]
    return out
