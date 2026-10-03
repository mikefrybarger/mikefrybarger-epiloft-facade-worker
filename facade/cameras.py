"""OpenSfM cameras and shots, projected the way OpenSfM projects them.

Projection is reimplemented here (rather than importing OpenSfM, which is a
heavy C++ build) and pinned to OpenSfM's conventions:

* world -> camera: ``x_cam = R @ X + t`` with ``R = rodrigues(shot.rotation)``;
  the camera looks down +Z, image x right, image y down.
* "normalized" image coordinates are centred on the image and scaled by
  ``max(width, height)``; pixels are ``n * size + (dim / 2 - 0.5)``.
* lens models follow ``opensfm/src/geometry/camera_functions.h``.

``tests/test_cameras.py`` round-trips every supported model.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .geometry import rodrigues

SUPPORTED_MODELS = ("perspective", "brown", "fisheye", "fisheye_opencv", "radial", "simple_radial")


@dataclass
class Camera:
    id: str
    model: str
    width: int
    height: int
    params: dict
    raw: dict | None = None

    @classmethod
    def from_json(cls, cam_id: str, data: dict):
        model = str(data.get("projection_type", "perspective")).lower()
        if model not in SUPPORTED_MODELS:
            raise ValueError(
                f"camera {cam_id}: projection_type '{model}' is not supported "
                f"(supported: {', '.join(SUPPORTED_MODELS)})"
            )
        params = {k: float(v) for k, v in data.items()
                  if isinstance(v, (int, float)) and k not in ("width", "height")}
        return cls(cam_id, model, int(data["width"]), int(data["height"]), params, dict(data))

    def _p(self, key, default=0.0):
        return self.params.get(key, default)

    def valid_angle(self, image_w: int, image_h: int) -> float:
        """Largest ray angle (radians off the optical axis) this lens model can
        project faithfully into this image.

        Polynomial lens models are only fitted inside the photo. Past that they
        bend back on themselves: for one real DJI M4E calibration the model
        folds at 53 deg off-axis and sends rays at 63.5 deg exactly onto the
        image centre, so wall points far outside the photo were "seen" in it.
        Valid = up to just past the image corners, and never past the fold.
        """
        key = (image_w, image_h)
        cache = self.__dict__.setdefault("_valid_cache", {})
        if key in cache:
            return cache[key]
        size = max(image_w, image_h)
        corner = float(np.hypot(image_w / 2.0, image_h / 2.0) / size)
        cx, cy = self._p("c_x"), self._p("c_y")
        angles = np.radians(np.linspace(0.0, 179.0, 17901))
        best = 0.0
        np_err = np.seterr(invalid="ignore", over="ignore")
        for phi in np.radians(np.arange(0.0, 360.0, 15.0)):     # all directions
            dx, dy = np.cos(phi), np.sin(phi)
            xc = np.sin(angles) * dx
            yc = np.sin(angles) * dy
            zc = np.cos(angles)
            nx, ny = self.project_normalized(xc, yc, zc, clip=False)
            rr = np.hypot(nx - cx, ny - cy)
            rr = np.where(np.isfinite(rr), rr, np.inf)
            grow = np.diff(rr) > 0
            fold = np.argmin(grow) if not grow.all() else len(rr) - 1   # first non-increase
            limit = fold
            past = np.flatnonzero(rr[:fold + 1] > corner * 1.08)
            if len(past):
                limit = min(limit, past[0])
            best = max(best, float(angles[max(limit, 1)]))
        np.seterr(**np_err)
        best = min(best, np.radians(179.0))
        cache[key] = best
        return best

    def focal_px(self, image_w: int, image_h: int) -> float:
        """Approximate focal length in pixels at the given image size."""
        size = max(image_w, image_h)
        if self.model in ("perspective", "fisheye"):
            f = self._p("focal")
        else:
            f = 0.5 * (self._p("focal_x", self._p("focal")) + self._p("focal_y", self._p("focal")))
        return f * size

    def project_normalized(self, xc: np.ndarray, yc: np.ndarray, zc: np.ndarray,
                           clip: bool = True, max_angle: float | None = None):
        """Camera-frame points -> OpenSfM normalized image coords (nx, ny).

        Points with zc <= 0 return NaN, and so do points further off-axis than
        max_angle (the lens model's valid range) when clip is set.
        """
        with np.errstate(divide="ignore", invalid="ignore"):
            behind = zc <= 1e-9
            if clip and max_angle is not None and self.model not in ("fisheye", "fisheye_opencv"):
                behind = behind | (np.arctan2(np.hypot(xc, yc), zc) > max_angle)
            elif clip and max_angle is not None:
                behind = np.arctan2(np.hypot(xc, yc), zc) > max_angle
            m = self.model
            if m in ("fisheye", "fisheye_opencv"):
                r = np.hypot(xc, yc)
                theta = np.arctan2(r, zc)
                scale = np.where(r > 1e-12, theta / np.maximum(r, 1e-12), 1.0 / np.maximum(zc, 1e-12))
                x, y = xc * scale, yc * scale
                t2 = theta * theta
                if m == "fisheye":
                    d = 1.0 + self._p("k1") * t2 + self._p("k2") * t2 * t2
                    f = self._p("focal")
                    nx, ny = f * d * x, f * d * y
                else:
                    d = 1.0 + t2 * (self._p("k1") + t2 * (self._p("k2") + t2 * (self._p("k3") + t2 * self._p("k4"))))
                    nx = self._p("focal_x") * d * x + self._p("c_x")
                    ny = self._p("focal_y") * d * y + self._p("c_y")
            else:
                x, y = xc / zc, yc / zc
                r2 = x * x + y * y
                if m == "perspective":
                    d = 1.0 + r2 * (self._p("k1") + self._p("k2") * r2)
                    f = self._p("focal")
                    nx, ny = f * d * x, f * d * y
                elif m == "brown":
                    k1, k2, k3 = self._p("k1"), self._p("k2"), self._p("k3")
                    p1, p2 = self._p("p1"), self._p("p2")
                    radial = 1.0 + r2 * (k1 + r2 * (k2 + r2 * k3))
                    xd = x * radial + 2.0 * p1 * x * y + p2 * (r2 + 2.0 * x * x)
                    yd = y * radial + 2.0 * p2 * x * y + p1 * (r2 + 2.0 * y * y)
                    nx = self._p("focal_x") * xd + self._p("c_x")
                    ny = self._p("focal_y") * yd + self._p("c_y")
                elif m == "radial":
                    d = 1.0 + r2 * (self._p("k1") + self._p("k2") * r2)
                    nx = self._p("focal_x") * d * x + self._p("c_x")
                    ny = self._p("focal_y") * d * y + self._p("c_y")
                else:  # simple_radial
                    d = 1.0 + self._p("k1") * r2
                    nx = self._p("focal_x") * d * x + self._p("c_x")
                    ny = self._p("focal_y") * d * y + self._p("c_y")
            nx = np.where(behind, np.nan, nx)
            ny = np.where(behind, np.nan, ny)
        return nx, ny


@dataclass
class Shot:
    name: str
    camera: Camera
    R: np.ndarray
    t: np.ndarray
    image_path: Path | None = None
    image_w: int = 0
    image_h: int = 0

    @property
    def center(self) -> np.ndarray:
        return -self.R.T @ self.t

    @property
    def optical_axis(self) -> np.ndarray:
        return self.R.T @ np.array([0.0, 0.0, 1.0])

    def size(self):
        w = self.image_w or self.camera.width
        h = self.image_h or self.camera.height
        return w, h

    def to_camera(self, points: np.ndarray):
        """World points (..., 3) -> camera-frame components (xc, yc, zc)."""
        pc = points @ self.R.T + self.t
        return pc[..., 0], pc[..., 1], pc[..., 2]

    def project(self, points: np.ndarray):
        """World points (..., 3) -> (px, py, depth) in pixels of the actual image.

        depth is the camera-frame Z. Points behind the camera get NaN pixels.
        """
        xc, yc, zc = self.to_camera(points)
        w, h = self.size()
        nx, ny = self.camera.project_normalized(xc, yc, zc, max_angle=self.camera.valid_angle(w, h))
        size = max(w, h)
        return nx * size + (w / 2.0 - 0.5), ny * size + (h / 2.0 - 0.5), zc

    def native_gsd(self, depth: np.ndarray) -> np.ndarray:
        """Metres per pixel on a surface facing the camera at this depth."""
        w, h = self.size()
        return np.asarray(depth) / self.camera.focal_px(w, h)


def load_reconstruction(path: Path, with_ids: bool = False):
    """First reconstruction in reconstruction.json -> (shots, sparse points (N,3) or None).

    with_ids=True also returns the sparse point ids (OpenSfM track ids)."""
    data = json.loads(Path(path).read_text())
    if isinstance(data, dict):
        data = [data]
    if not data:
        raise RuntimeError("reconstruction.json is empty")
    recon = data[0]
    cameras = {cid: Camera.from_json(cid, c) for cid, c in (recon.get("cameras") or {}).items()}
    shots = {}
    for name, s in (recon.get("shots") or {}).items():
        cam_id = s.get("camera")
        if cam_id not in cameras:
            raise RuntimeError(f"shot {name} references unknown camera {cam_id}")
        shots[name] = Shot(
            name=name,
            camera=cameras[cam_id],
            R=rodrigues(s["rotation"]),
            t=np.asarray(s["translation"], dtype=np.float64),
        )
    if not shots:
        raise RuntimeError("reconstruction.json contains no shots")
    if len(data) > 1:
        others = sum(len(r.get("shots") or {}) for r in data[1:])
        print(f"note: reconstruction.json has {len(data)} partial reconstructions; "
              f"using the first ({len(shots)} shots, {others} shots in the others)", flush=True)
    ids, pts = [], []
    for pid, p in (recon.get("points") or {}).items():
        c = p.get("coordinates")
        if c and len(c) == 3:
            ids.append(str(pid))
            pts.append(c)
    pts = np.asarray(pts, dtype=np.float64).reshape(-1, 3)
    sparse = pts if len(pts) else None
    if with_ids:
        return shots, sparse, ids
    return shots, sparse
