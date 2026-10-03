import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from facade.cameras import Camera, Shot  # noqa: E402
from facade.geometry import FrameContext, OrthoGrid, WallPlane, rodrigues  # noqa: E402
from facade.pointcloud import read_ply_xyz  # noqa: E402

CTX = FrameContext.from_payload((643844.0, 4884561.0), {"e": 643850.0, "n": 4884555.0, "z": 1.0})


def splat_worker_splat_to_mesh(x, y, z):
    """Copied from epiloft-opensplat-worker splat_to_mesh (keep_crs, scale 1)."""
    east, north = x + 643844.0, y + 4884561.0
    return [east - 643850.0, z - 1.0, -(north - 4884555.0)]


def test_mesh_frame_matches_splat_worker_contract():
    p = np.array([12.3, -4.5, 6.7])
    assert np.allclose(CTX.opensfm_to_mesh(p), splat_worker_splat_to_mesh(*p))
    assert np.allclose(CTX.mesh_to_opensfm(CTX.opensfm_to_mesh(p)), p)


def test_plane_frame_is_right_handed_and_orthonormal():
    pl = WallPlane.from_corners([0, 0, 0], [4, 0, 0.0], [0.1, 0, 3])  # slightly leaning pick
    assert np.isclose(pl.height_m, 3.0)
    assert np.allclose(np.cross(pl.u, pl.v), pl.w)
    assert np.isclose(pl.u @ pl.v, 0)
    uvw = pl.to_wall(pl.to_world(1.5, 2.0, 0.25)[None])
    assert np.allclose(uvw, [[1.5, 2.0, 0.25]])


def test_plane_flip_keeps_left_to_right():
    pl = WallPlane.from_corners([0, 0, 0], [4, 0, 0], [0, 0, 3])  # W = -y
    cams = np.array([[2.0, 5.0, 1.5]])                             # cameras on +y
    axes = np.array([[0.0, -1.0, 0.0]])
    assert pl.face_cameras(cams, axes)
    assert np.allclose(pl.w, [0, 1, 0])
    assert np.allclose(np.cross(pl.u, pl.v), pl.w)
    assert np.allclose(pl.origin, [4, 0, 0])


def test_degenerate_corners_rejected():
    with pytest.raises(ValueError):
        WallPlane.from_corners([0, 0, 0], [0.01, 0, 0], [0, 0, 3])
    with pytest.raises(ValueError):
        WallPlane.from_corners([0, 0, 0], [4, 0, 0], [2, 0, 0])


def test_grid_pixel_convention():
    g = OrthoGrid(gsd_m=0.01, width_px=100, height_px=50, height_m=0.5)
    u, v = g.uv(0, 1, 0, 1)
    assert np.isclose(u[0, 0], 0.005) and np.isclose(v[0, 0], 0.495)
    assert g.image_xy_to_uv(100, 50) == (1.0, 0.0)


@pytest.mark.parametrize("model,params", [
    ("perspective", {"focal": 0.9, "k1": -0.05, "k2": 0.01}),
    ("brown", {"focal_x": 0.8, "focal_y": 0.81, "c_x": 0.01, "c_y": -0.02,
               "k1": -0.1, "k2": 0.02, "k3": 0.001, "p1": 0.001, "p2": -0.002}),
    ("fisheye", {"focal": 0.5, "k1": 0.01, "k2": -0.002}),
    ("fisheye_opencv", {"focal_x": 0.5, "focal_y": 0.5, "c_x": 0, "c_y": 0,
                        "k1": 0.01, "k2": 0, "k3": 0, "k4": 0}),
    ("radial", {"focal_x": 0.8, "focal_y": 0.8, "c_x": 0, "c_y": 0, "k1": -0.05, "k2": 0.0}),
    ("simple_radial", {"focal_x": 0.8, "focal_y": 0.8, "c_x": 0, "c_y": 0, "k1": -0.05}),
])
def test_camera_models_project_centre_and_are_monotonic(model, params):
    cam = Camera.from_json("c", {"projection_type": model, "width": 4000, "height": 3000, **params})
    shot = Shot("s", cam, np.eye(3), np.zeros(3))
    px, py, d = shot.project(np.array([[0.0, 0.0, 10.0]]))
    cx = params.get("c_x", 0) * 4000
    cy = params.get("c_y", 0) * 4000
    assert np.isclose(px[0], 1999.5 + cx) and np.isclose(py[0], 1499.5 + cy) and d[0] == 10
    xs = np.linspace(0, 4, 20)
    pts = np.stack([xs, np.zeros_like(xs), np.full_like(xs, 10.0)], -1)
    px, _, _ = shot.project(pts)
    assert np.all(np.diff(px) > 0)
    behind, _, _ = shot.project(np.array([[0.0, 0.0, -5.0]]))
    assert np.isnan(behind[0])


def test_unknown_model_is_rejected():
    with pytest.raises(ValueError, match="not supported"):
        Camera.from_json("c", {"projection_type": "spherical", "width": 10, "height": 10})


def test_shot_center_from_opensfm_pose():
    R = rodrigues([0.1, -0.2, 0.3])
    C = np.array([5.0, 6.0, 7.0])
    cam = Camera.from_json("c", {"projection_type": "perspective", "width": 10, "height": 10, "focal": 1})
    shot = Shot("s", cam, R, -R @ C)
    assert np.allclose(shot.center, C)


def test_ply_reader(tmp_path):
    pts = np.random.default_rng(0).normal(size=(50, 3)).astype("<f4")
    p = tmp_path / "a.ply"
    hdr = ("ply\nformat binary_little_endian 1.0\nelement vertex 50\nproperty float x\n"
           "property float y\nproperty float z\nproperty uchar red\nproperty uchar green\n"
           "property uchar blue\nend_header\n").encode()
    rec = np.zeros(50, dtype=[("x", "<f4"), ("y", "<f4"), ("z", "<f4"), ("r", "u1"), ("g", "u1"), ("b", "u1")])
    rec["x"], rec["y"], rec["z"] = pts[:, 0], pts[:, 1], pts[:, 2]
    p.write_bytes(hdr + rec.tobytes())
    assert np.allclose(read_ply_xyz(p), pts)
