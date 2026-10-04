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


def test_depth_map_with_holes_stays_in_band():
    """Regression: scattered holes (glass, sparse areas) must never produce
    depths outside what was measured, let alone outside the search band."""
    from facade.depth import build_depth_map

    rng = np.random.default_rng(0)
    u = rng.uniform(0, 10, 200_000)
    v = rng.uniform(0, 4, 200_000)
    w = np.where(u < 5, 0.2, -0.3) + rng.normal(0, 0.005, u.size)   # a step in the wall
    keep = rng.random(u.size) > 0.35                                 # scattered dropout
    keep &= ~((u > 2) & (u < 3) & (v > 1) & (v < 2.5))               # a window with no points
    pts = np.stack([u[keep], v[keep], w[keep]], -1)
    d = build_depth_map(pts, 10, 4, cell_m=0.02, depth_front_m=0.6, depth_back_m=0.5)
    assert d.coverage < 0.95                                         # holes really exist
    assert d.grid.min() >= -0.32 and d.grid.max() <= 0.22, (d.grid.min(), d.grid.max())
    # the window is filled from its surroundings, not invented
    win = d.sample(np.array([2.5]), np.array([1.75]))[0]
    assert abs(win - 0.2) < 0.02


def test_m4e_lens_model_never_folds_back_into_the_image():
    """Real DJI M4E calibration: the polynomial folds at 53 deg and maps 63.5 deg
    onto the image centre. Only rays inside the photo may project."""
    raw = {"projection_type": "brown", "width": 5280, "height": 3956,
           "focal_x": 0.7052558642285107, "focal_y": 0.7052558642285107,
           "c_x": 0.004969223516840049, "c_y": -0.00470324524625981, "k1": -0.10341031555259611,
           "k2": -0.01531662859893411, "p1": 2.6876150908751606e-06,
           "p2": -0.00020501113013549115, "k3": -0.005277757045775089}
    cam = Camera.from_json("m4e", raw)
    shot = Shot("s", cam, np.eye(3), np.zeros(3), None, 5280, 3956)
    assert 45 < np.degrees(cam.valid_angle(5280, 3956)) < 52
    ang = np.radians(np.linspace(0, 89, 400))
    for phi in np.radians([0, 37, 90, 145, 210, 300]):
        pts = np.stack([np.sin(ang) * np.cos(phi), np.sin(ang) * np.sin(phi), np.cos(ang)], -1)
        px, py, _ = shot.project(pts)
        ok = np.isfinite(px)
        pp_x = 2639.5 + raw["c_x"] * 5280                 # principal point, pixels
        pp_y = 1977.5 + raw["c_y"] * 5280
        r = np.hypot(px[ok] - pp_x, py[ok] - pp_y)
        assert np.all(np.diff(r) > 0)                     # strictly outward: no fold
        assert not np.any(ok & (np.degrees(ang) > 52))    # nothing from outside the field


def test_windows_stay_on_the_wall_and_posts_are_obstacles():
    """Glass: sparse interior points behind it, must stay on the wall plane.
    Post in front with wall visible behind it: an obstacle, not facade.
    Sign: solid, no wall behind it: keeps its own depth."""
    from facade.depth import build_depth_map

    rng = np.random.default_rng(1)
    n = 300_000
    u, v = rng.uniform(0, 10, n), rng.uniform(0, 4, n)
    w = rng.normal(0, 0.004, n)
    glass = (u > 2) & (u < 4) & (v > 0.8) & (v < 2.5)
    sign = (u > 6) & (u < 9) & (v > 3.0) & (v < 3.6)
    keep = ~glass | (rng.random(n) < 0.08)                       # few points through glass
    w = np.where(glass, -0.6 + rng.normal(0, 0.1, n), w)         # ...from the shop interior
    w = np.where(sign, 0.25, w)                                  # sign face 25 cm proud
    pts = [np.stack([u[keep], v[keep], w[keep]], -1)]
    pu, pv = rng.uniform(4.9, 5.1, 20000), rng.uniform(0, 4, 20000)
    pts.append(np.stack([pu, pv, np.full_like(pu, 0.9)], -1))  # post 0.9 m in front
    d = build_depth_map(np.concatenate(pts), 10, 4, cell_m=0.02, depth_front_m=1.2, depth_back_m=1.0)
    s = lambda uu, vv: d.sample(np.array([uu]), np.array([vv]))[0]   # noqa: E731
    assert abs(s(3.0, 1.6)) < 0.02, s(3.0, 1.6)      # window: wall plane
    assert abs(s(5.0, 1.5)) < 0.02, s(5.0, 1.5)      # behind the post: wall plane
    assert abs(s(7.5, 3.3) - 0.25) < 0.03, s(7.5, 3.3)   # sign keeps its depth


def test_remap_handles_maps_past_opencvs_size_limit():
    """Regression: the refinement passed a 1-row map of every point on a 49 m
    wall to cv2.remap, which rejects any side >= 32,767."""
    from facade.images import remap

    img = (np.arange(200 * 300, dtype=np.float32) % 251).reshape(200, 300)
    rng = np.random.default_rng(0)
    n = 100_003
    xs, ys = rng.integers(0, 300, n), rng.integers(0, 200, n)
    out = remap(img, xs.astype(np.float32), ys.astype(np.float32))
    assert out.shape == (n,) and np.array_equal(out, img[ys, xs])
    col = remap(np.dstack([img] * 3), xs.astype(np.float32)[None], ys.astype(np.float32)[None])
    assert col.shape == (1, n, 3) and np.array_equal(col[0, :, 1], img[ys, xs])
