"""Wall plane frame and coordinate transforms.

Frames used in this worker:

* **opensfm**: the frame of ``opensfm/reconstruction.json``. For a
  georeferenced ODM run this is UTM minus the ``coords.txt`` offset, Z up,
  metres (``source_frame = odm_utm_offset`` in the splat worker's alignment
  contract).
* **mesh**: the three.js (Y-up) frame Studio renders in. The splat worker's
  ``splat_to_mesh`` defines it as ``[E - origin.e, Z - origin.z, -(N - origin.n)]``
  where ``origin`` is the run's ``mesh_origin``. This module implements the
  inverse so a wall picked in Studio lands exactly on the photos.
* **wall**: the local orthoplane. ``U`` runs left to right along the wall as
  seen from outside, ``V`` runs up, ``W`` is the outward normal (toward the
  cameras). ``U x V = W``. The plane origin is the bottom-left corner.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


def rodrigues(rvec) -> np.ndarray:
    """Rotation matrix from an OpenSfM axis-angle vector."""
    r = np.asarray(rvec, dtype=np.float64).reshape(3)
    theta = float(np.linalg.norm(r))
    if theta < 1e-12:
        return np.eye(3)
    k = r / theta
    kx = np.array([[0.0, -k[2], k[1]], [k[2], 0.0, -k[0]], [-k[1], k[0], 0.0]])
    return np.eye(3) + np.sin(theta) * kx + (1.0 - np.cos(theta)) * (kx @ kx)


# ---------------------------------------------------------------------------
# mesh (three.js, Y up) <-> opensfm (UTM offset, Z up)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class FrameContext:
    """Everything needed to move points between the mesh and OpenSfM frames.

    offset_e / offset_n come from ``odm_georeferencing/coords.txt``.
    mesh_origin is the run's ``mesh_origin`` ({e, n, z}) used by Studio.
    """
    offset_e: float
    offset_n: float
    mesh_origin_e: float
    mesh_origin_n: float
    mesh_origin_z: float

    @classmethod
    def from_payload(cls, coords_offset, mesh_origin):
        return cls(
            offset_e=float(coords_offset[0]),
            offset_n=float(coords_offset[1]),
            mesh_origin_e=float(mesh_origin["e"]),
            mesh_origin_n=float(mesh_origin["n"]),
            mesh_origin_z=float(mesh_origin.get("z", 0.0)),
        )

    def mesh_to_opensfm(self, p) -> np.ndarray:
        p = np.asarray(p, dtype=np.float64)
        mx, my, mz = p[..., 0], p[..., 1], p[..., 2]
        east = mx + self.mesh_origin_e
        north = -mz + self.mesh_origin_n
        up = my + self.mesh_origin_z
        return np.stack([east - self.offset_e, north - self.offset_n, up], axis=-1)

    def opensfm_to_mesh(self, p) -> np.ndarray:
        p = np.asarray(p, dtype=np.float64)
        east = p[..., 0] + self.offset_e
        north = p[..., 1] + self.offset_n
        up = p[..., 2]
        return np.stack(
            [east - self.mesh_origin_e, up - self.mesh_origin_z, -(north - self.mesh_origin_n)],
            axis=-1,
        )


# ---------------------------------------------------------------------------
# Wall plane
# ---------------------------------------------------------------------------

@dataclass
class WallPlane:
    origin: np.ndarray       # bottom-left corner, opensfm frame
    u: np.ndarray            # unit, along the wall (viewer's right)
    v: np.ndarray            # unit, up the wall
    w: np.ndarray            # unit, outward normal toward the cameras
    width_m: float
    height_m: float
    notes: list = field(default_factory=list)
    view_from: np.ndarray | None = None   # Studio camera position when the wall was picked

    @classmethod
    def from_corners(cls, bottom_left, bottom_right, top_left):
        bl = np.asarray(bottom_left, dtype=np.float64)
        br = np.asarray(bottom_right, dtype=np.float64)
        tl = np.asarray(top_left, dtype=np.float64)
        along = br - bl
        width = float(np.linalg.norm(along))
        if width < 0.05:
            raise ValueError("wall baseline is shorter than 5 cm; check the picked corners")
        u = along / width
        up = tl - bl
        up = up - np.dot(up, u) * u  # make V exactly perpendicular to U
        height = float(np.linalg.norm(up))
        if height < 0.05:
            raise ValueError("wall height is under 5 cm or the top corner is on the baseline")
        v = up / height
        w = np.cross(u, v)
        return cls(origin=bl, u=u, v=v, w=w / np.linalg.norm(w), width_m=width, height_m=height)

    def face_cameras(self, camera_centers: np.ndarray, optical_axes: np.ndarray) -> bool:
        """Point W toward the cameras that look at this wall. True if flipped.

        Only photos aimed at the plane vote (in front of it and looking back
        at it), so nadir shots over the roof do not decide which side is out.
        Flipping W alone would mirror the image, so U flips too and the origin
        moves to the other bottom corner. The result still reads left to right
        for someone standing outside looking at the wall.
        """
        if len(camera_centers) == 0:
            return False
        side = (np.asarray(camera_centers) - self.origin) @ self.w
        facing = np.asarray(optical_axes) @ self.w
        votes_plus = int(np.sum((side > 0) & (facing < -0.3)))
        votes_minus = int(np.sum((side < 0) & (facing > 0.3)))
        if votes_plus >= votes_minus:
            return False
        self.flip("plane flipped so its normal faces the cameras")
        return True

    def flip(self, note: str):
        """Turn the plane around: W points the other way, U too (so the image
        still reads left to right from the new outside), origin moves to the
        other bottom corner."""
        self.origin = self.origin + self.width_m * self.u
        self.u = -self.u
        self.w = -self.w
        self.notes.append(note)

    def orient_toward(self, point) -> bool:
        """Make W point at `point` (e.g. where the user stood in Studio). True if flipped."""
        if (np.asarray(point, dtype=np.float64) - self.origin) @ self.w >= 0:
            return False
        self.flip("plane turned to face the side it was picked from in Studio")
        return True

    def to_wall(self, points: np.ndarray) -> np.ndarray:
        """opensfm points (N,3) -> wall coords (N,3) as (u, v, w) in metres."""
        d = np.asarray(points, dtype=np.float64) - self.origin
        return np.stack([d @ self.u, d @ self.v, d @ self.w], axis=-1)

    def to_world(self, u, v, w) -> np.ndarray:
        """Wall coords (broadcastable arrays) -> opensfm points (..., 3)."""
        u = np.asarray(u, dtype=np.float64)[..., None]
        v = np.asarray(v, dtype=np.float64)[..., None]
        w = np.asarray(w, dtype=np.float64)[..., None]
        return self.origin + u * self.u + v * self.v + w * self.w

    def corners(self) -> dict:
        o = self.origin
        return {
            "bottom_left": o,
            "bottom_right": o + self.width_m * self.u,
            "top_left": o + self.height_m * self.v,
            "top_right": o + self.width_m * self.u + self.height_m * self.v,
        }

    def describe(self, frame: FrameContext | None = None) -> dict:
        def lst(a):
            return [round(float(x), 6) for x in a]

        corners = self.corners()
        out = {
            "width_m": round(self.width_m, 6),
            "height_m": round(self.height_m, 6),
            "opensfm": {
                "origin": lst(self.origin),
                "u": lst(self.u), "v": lst(self.v), "w": lst(self.w),
                "corners": {k: lst(c) for k, c in corners.items()},
            },
            "notes": list(self.notes),
        }
        if frame is not None:
            out["mesh"] = {k: lst(frame.opensfm_to_mesh(c)) for k, c in corners.items()}
        return out


@dataclass(frozen=True)
class OrthoGrid:
    """Pixel grid on the wall. Row 0 is the top of the wall."""
    gsd_m: float
    width_px: int
    height_px: int
    height_m: float

    @classmethod
    def for_plane(cls, plane: WallPlane, gsd_m: float):
        return cls(
            gsd_m=gsd_m,
            width_px=max(1, int(np.ceil(plane.width_m / gsd_m))),
            height_px=max(1, int(np.ceil(plane.height_m / gsd_m))),
            height_m=plane.height_m,
        )

    def uv(self, row0: int, row1: int, col0: int, col1: int):
        """Wall (u, v) of pixel centres for rows [row0,row1) x cols [col0,col1).

        Indices may fall outside the image (tile margins); the maths is the same.
        """
        cols = (np.arange(col0, col1, dtype=np.float64) + 0.5) * self.gsd_m
        rows = self.height_m - (np.arange(row0, row1, dtype=np.float64) + 0.5) * self.gsd_m
        return np.meshgrid(cols, rows)  # u (H,W), v (H,W)

    def image_xy_to_uv(self, x: float, y: float):
        """Continuous image coords (0,0 = top-left corner of the image, the
        convention most viewers report) -> wall (u, v) in metres."""
        return x * self.gsd_m, self.height_m - y * self.gsd_m
