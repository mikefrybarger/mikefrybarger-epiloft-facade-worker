"""Decoded photo cache with pyramid levels and a memory budget."""
from __future__ import annotations

from collections import OrderedDict

import cv2
import numpy as np

# Match OpenSfM: it decodes without applying EXIF rotation, so poses refer to
# the stored pixel layout. Applying the rotation would misalign every photo
# taken with the camera turned.
READ_FLAGS = cv2.IMREAD_COLOR | getattr(cv2, "IMREAD_IGNORE_ORIENTATION", 0)
REDUCED_FLAGS = {
    2: cv2.IMREAD_REDUCED_COLOR_2, 4: cv2.IMREAD_REDUCED_COLOR_4, 8: cv2.IMREAD_REDUCED_COLOR_8,
}


def read_image(path, reduce: int = 1) -> np.ndarray:
    flags = REDUCED_FLAGS.get(reduce, cv2.IMREAD_COLOR) | getattr(cv2, "IMREAD_IGNORE_ORIENTATION", 0)
    img = cv2.imread(str(path), flags if reduce > 1 else READ_FLAGS)
    if img is None:
        raise RuntimeError(f"could not decode image {path}")
    return img


def image_size(path):
    """(width, height) of the stored pixels, without EXIF rotation.

    Pillow reads this from the header, so no full decode is needed.
    """
    try:
        from PIL import Image  # noqa: PLC0415

        with Image.open(path) as im:
            return im.size
    except ImportError:
        h, w = read_image(path).shape[:2]
        return w, h


class ImageCache:
    """LRU of decoded photos. Level n is the photo downsampled by 2**n."""

    def __init__(self, budget_bytes: int):
        self.budget = budget_bytes
        self.used = 0
        self._items: OrderedDict = OrderedDict()
        self.decodes = 0

    def get(self, shot, level: int = 0) -> np.ndarray:
        key = (shot.name, level)
        if key in self._items:
            self._items.move_to_end(key)
            return self._items[key]
        if level == 0:
            img = read_image(shot.image_path)
            self.decodes += 1
        else:
            img = cv2.pyrDown(self.get(shot, level - 1))
        self._items[key] = img
        self.used += img.nbytes
        while self.used > self.budget and len(self._items) > 1:
            _, old = self._items.popitem(last=False)
            self.used -= old.nbytes
        return img


REMAP_MAX = 32000          # OpenCV remap needs every map and image side < 32767


def remap(src: np.ndarray, mx: np.ndarray, my: np.ndarray, interp=cv2.INTER_LINEAR,
          border=cv2.BORDER_REPLICATE) -> np.ndarray:
    """cv2.remap for maps of any shape or length.

    OpenCV refuses maps with a side of 32,767 or more; a 1-row map of every
    sample point on a 49 m wall is far longer. Maps are reshaped into rows of
    at most 4096 and processed in blocks, then put back in the input shape.
    """
    if src.shape[0] >= REMAP_MAX or src.shape[1] >= REMAP_MAX:
        raise ValueError(f"remap source {src.shape[:2]} exceeds OpenCV's 32,767 px limit")
    mx = np.asarray(mx, dtype=np.float32)
    my = np.asarray(my, dtype=np.float32)
    shape = mx.shape
    if mx.ndim == 2 and max(shape) < REMAP_MAX:
        return cv2.remap(src, mx, my, interp, borderMode=border)
    flat_x, flat_y = mx.ravel(), my.ravel()
    n = flat_x.size
    width = 4096
    rows = -(-n // width)
    pad = rows * width - n
    fx = np.concatenate([flat_x, np.full(pad, -1, np.float32)]).reshape(rows, width)
    fy = np.concatenate([flat_y, np.full(pad, -1, np.float32)]).reshape(rows, width)
    parts = []
    for r0 in range(0, rows, 4096):
        parts.append(cv2.remap(src, fx[r0:r0 + 4096], fy[r0:r0 + 4096], interp, borderMode=border))
    out = np.concatenate(parts, axis=0)
    out = out.reshape((rows * width,) + out.shape[2:])[:n]
    return out.reshape(shape + out.shape[1:])
