"""A synthetic ODM project: a textured wall, a red post in front of it, and
photos rendered by ray tracing through real OpenSfM lens models.

Because the wall texture is a known function, the facade output can be
checked pixel for pixel against ground truth.
"""
from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np

WALL_W, WALL_H = 6.0, 3.0
YAW = np.radians(30.0)
ORIGIN = np.array([12.0, -4.0, 1.5])            # bottom-left corner, opensfm frame
U = np.array([np.cos(YAW), np.sin(YAW), 0.0])
V = np.array([0.0, 0.0, 1.0])
W = np.cross(U, V)                               # outward normal
POST_U, POST_W, POST_R = 3.1, 1.3, 0.09          # magenta post in front of the wall
OFFSET_E, OFFSET_N = 643_844.0, 4_884_561.0
MESH_ORIGIN = {"e": 643_850.0, "n": 4_884_555.0, "z": 1.0}
IMG_W, IMG_H = 1200, 900
CAMERA = {
    "projection_type": "brown", "width": IMG_W, "height": IMG_H,
    "focal_x": 0.80, "focal_y": 0.80, "c_x": 0.004, "c_y": -0.003,
    "k1": -0.06, "k2": 0.02, "k3": 0.0, "p1": 0.0008, "p2": -0.0005,
}
SKY = np.array([200.0, 190.0, 170.0])  # BGR


def wall_texture(u, v):
    """BGR colour of the wall at (u, v). Checker of 0.25 m plus fine stripes."""
    cu, cv_ = np.floor(u / 0.25).astype(int), np.floor(v / 0.25).astype(int)
    checker = ((cu + cv_) % 2).astype(np.float64)
    hue = (cu * 7 + cv_ * 3) % 5
    palette = np.array([[60, 90, 160], [150, 120, 70], [80, 160, 90], [170, 170, 170], [90, 70, 130]], float)
    base = palette[hue] * (0.6 + 0.4 * checker[..., None])
    stripes = 0.5 + 0.5 * np.sin(2 * np.pi * v / 0.04)
    return base * (0.85 + 0.15 * stripes[..., None])


def world_wall(u, v, w=0.0):
    return ORIGIN + np.asarray(u)[..., None] * U + np.asarray(v)[..., None] * V + np.asarray(w)[..., None] * W


def look_at(center, target):
    z = target - center
    z /= np.linalg.norm(z)
    x = np.cross(z, np.array([0.0, 0.0, 1.0]))
    x /= np.linalg.norm(x)
    y = np.cross(z, x)
    return np.stack([x, y, z])  # rows: camera axes in world


def undistort_brown(xd, yd, c):
    x, y = xd.copy(), yd.copy()
    for _ in range(20):
        r2 = x * x + y * y
        radial = 1 + r2 * (c["k1"] + r2 * (c["k2"] + r2 * c["k3"]))
        dx = 2 * c["p1"] * x * y + c["p2"] * (r2 + 2 * x * x)
        dy = 2 * c["p2"] * x * y + c["p1"] * (r2 + 2 * y * y)
        x, y = (xd - dx) / radial, (yd - dy) / radial
    return x, y


STOREFRONT = {"enabled": False}
STORE_V, STORE_W = 1.9, -0.25                       # glass line 25 cm behind the band above
WINDOWS = [(0.4, 2.6, 0.3, 1.7), (3.4, 5.6, 0.3, 1.7)]   # u0, u1, v0, v1


def in_window(u, v):
    m = np.zeros(np.shape(u), dtype=bool)
    for u0, u1, v0, v1 in WINDOWS:
        m |= (u > u0) & (u < u1) & (v > v0) & (v < v1)
    return m


def render(R, C, gain, shade=None):
    size = max(IMG_W, IMG_H)
    pxs, pys = np.meshgrid(np.arange(IMG_W, dtype=float), np.arange(IMG_H, dtype=float))
    nx = (pxs - (IMG_W / 2 - 0.5)) / size
    ny = (pys - (IMG_H / 2 - 0.5)) / size
    xd = (nx - CAMERA["c_x"]) / CAMERA["focal_x"]
    yd = (ny - CAMERA["c_y"]) / CAMERA["focal_y"]
    x, y = undistort_brown(xd, yd, CAMERA)
    rays = np.stack([x, y, np.ones_like(x)], -1) @ R  # camera -> world directions
    # wall plane
    rel = C - ORIGIN
    denom = rays @ W
    t_wall = -(rel @ W) / np.where(np.abs(denom) < 1e-9, 1e-9, denom)
    hit = C + t_wall[..., None] * rays
    hu, hv = (hit - ORIGIN) @ U, (hit - ORIGIN) @ V
    if STOREFRONT["enabled"]:   # lower storey set back: glass line behind the band above
        t_low = -(rel @ W - STORE_W) / np.where(np.abs(denom) < 1e-9, 1e-9, denom)
        hit_l = C + t_low[..., None] * rays
        lu, lv = (hit_l - ORIGIN) @ U, (hit_l - ORIGIN) @ V
        use_low = (hv < STORE_V) & (lv < STORE_V)
        t_wall = np.where(use_low, t_low, t_wall)
        hu, hv = np.where(use_low, lu, hu), np.where(use_low, lv, hv)
    on_wall = (t_wall > 0) & (hu >= -0.5) & (hu <= WALL_W + 0.5) & (hv >= -0.5) & (hv <= WALL_H + 0.5)
    img = np.where(on_wall[..., None], wall_texture(hu, hv), SKY)
    if STOREFRONT["enabled"]:   # glass reflects something different from every camera
        refl = 18.0 * np.sin(0.9 * C[0] + 0.6 * C[1] + 2.0 * hu)
        img = np.where((on_wall & in_window(hu, hv))[..., None], img + refl[..., None], img)
    # vertical post (infinite cylinder clipped to wall height + 0.5)
    post_c = ORIGIN + POST_U * U + POST_W * W
    d2 = rays[..., :2]
    o2 = (C - post_c)[:2]
    a = (d2 ** 2).sum(-1)
    b = 2 * (d2 @ o2)
    cc = o2 @ o2 - POST_R ** 2
    disc = b * b - 4 * a * cc
    t_post = np.where(disc >= 0, (-b - np.sqrt(np.maximum(disc, 0))) / (2 * a), np.inf)
    zhit = C[2] + t_post * rays[..., 2]
    post = (disc >= 0) & (t_post > 0) & (zhit >= ORIGIN[2] - 0.2) & (zhit <= ORIGIN[2] + WALL_H + 0.5)
    post &= ~on_wall | (t_post < t_wall)
    img = np.where(post[..., None], np.array([235.0, 20.0, 235.0]), img)  # magenta: no wall colour is close
    if shade is not None:  # brightness drifting across the frame: lens falloff + a sun-angle gradient
        a, b, tint = shade
        xn, yn = pxs / IMG_W - 0.5, pys / IMG_H - 0.5
        field = (1 + a * xn + b * yn) * (1 - 0.35 * (xn * xn + yn * yn) * 2)
        img = img * field[..., None] * np.asarray(tint)[None, None, :]
    return np.clip(img * gain, 0, 255).astype(np.uint8)


def camera_positions():
    """A facade pass: two rows of photos ~4 m off the wall, plus a nadir shot."""
    cams = []
    for row, hv in enumerate((0.9, 2.2)):
        for i, uu in enumerate(np.linspace(0.6, 5.4, 7)):
            c = world_wall(uu, hv, 4.0 + 0.2 * ((i + row) % 2))
            target = world_wall(uu + 0.2 * ((i % 3) - 1), hv, 0.0)
            cams.append((f"IMG_{row}{i:02d}.JPG", c, target))
    top = world_wall(3.0, WALL_H + 8.0, 2.0)
    cams.append(("IMG_NADIR.JPG", top, top - np.array([0.0, 0.0, 10.0]) + 0.01 * U))
    return cams


def write_ply(path: Path, pts: np.ndarray):
    pts = pts.astype("<f4")
    header = ("ply\nformat binary_little_endian 1.0\n"
              f"element vertex {len(pts)}\nproperty float x\nproperty float y\nproperty float z\n"
              "end_header\n").encode()
    path.write_bytes(header + pts.tobytes())


BUILDING_DEPTH = 8.0


def boundary_wobble(u, amp):
    """How far the cloud's band/glass edge wanders from the real (straight) one.

    Photogrammetry smears a depth step over a few cm and the 85th-percentile
    rasterisation picks up a different side of it cell by cell, so the edge
    in the cloud is wavy even though the building's is dead straight."""
    return amp * (0.6 * np.sin(2 * np.pi * u / 0.9) + 0.4 * np.sin(2 * np.pi * u / 0.37 + 1.0))


def smart3d_positions():
    """DJI Smart 3D style: almost every photo looks down at the facade from
    above at 20-40 degrees, with some yaw, instead of square-on rows."""
    cams = []
    for row, (hv, standoff, aim_v) in enumerate(((4.6, 4.5, 1.0), (3.4, 4.0, 1.2))):
        for i, uu in enumerate(np.linspace(0.4, WALL_W - 0.4, 8)):
            c = world_wall(uu, hv, standoff + 0.15 * (i % 2))
            target = world_wall(uu + 0.6 * ((i % 3) - 1), aim_v, 0.0)
            cams.append((f"S3D_{row}{i:02d}.JPG", c, target))
    return cams


def point_cloud(spacing=0.03, seed=0, post_ring=24, post_step=0.03, ground=False, dropout=0.0,
                building=False, details=False, wobble=0.0):
    rng = np.random.default_rng(seed)
    uu, vv = np.meshgrid(np.arange(0, WALL_W, spacing), np.arange(0, WALL_H, spacing))
    wall = world_wall(uu.ravel(), vv.ravel(), rng.normal(0, 0.004, uu.size))
    if dropout:  # glass, dark paint, sparse matching: scattered gaps plus a window
        uf, vf = uu.ravel(), vv.ravel()
        keep = rng.random(len(wall)) > dropout
        keep &= ~((uf > 1.0) & (uf < 2.2) & (vf > 1.0) & (vf < 2.2))
        wall = wall[keep]
    ang = np.linspace(0, 2 * np.pi, post_ring, endpoint=False)
    zs = np.arange(-0.2, WALL_H + 0.5, post_step)
    aa, zz = np.meshgrid(ang, zs)
    post_c = ORIGIN + POST_U * U + POST_W * W
    post = np.stack([post_c[0] + POST_R * np.cos(aa.ravel()), post_c[1] + POST_R * np.sin(aa.ravel()),
                     ORIGIN[2] + zz.ravel()], -1)
    if STOREFRONT["enabled"]:
        uf, vf = (wall - ORIGIN) @ U, (wall - ORIGIN) @ V
        low = vf < STORE_V + boundary_wobble(uf, wobble)
        wall = wall + np.where(low, STORE_W, 0.0)[:, None] * W
        glass = low & in_window(uf, vf)
        keep = ~glass | (rng.random(len(wall)) < 0.06)
        wall = np.where(glass[:, None], wall - 0.6 * W, wall)[keep]     # a few interior points
        su, sw = np.meshgrid(np.arange(0, WALL_W, spacing), np.arange(STORE_W, 0, spacing))
        sv = STORE_V + boundary_wobble(su.ravel(), wobble)
        wall = np.concatenate([wall, world_wall(su.ravel(), sv, sw.ravel())])
    parts = [wall, post]
    if details:  # sills and mullion frames standing a few cm off the wall
        for vs in (0.9, 2.1):
            su, sw = np.meshgrid(np.arange(0, WALL_W, 0.01), np.arange(0.01, 0.15, 0.01))
            parts.append(world_wall(su.ravel(), np.full(su.size, vs), sw.ravel()))
        for uf in np.arange(0.5, WALL_W, 1.0):
            fv, fw = np.meshgrid(np.arange(0, WALL_H, 0.01), np.arange(0.01, 0.07, 0.01))
            parts.append(world_wall(np.full(fv.size, uf), fv.ravel(), fw.ravel()))
    if building:  # the rest of the building behind the facade: back wall, roof, ends
        s2 = 0.06
        bu, bv = np.meshgrid(np.arange(0, WALL_W, s2), np.arange(0, WALL_H, s2))
        parts.append(world_wall(bu.ravel(), bv.ravel(), np.full(bu.size, -BUILDING_DEPTH)))
        ru, rw = np.meshgrid(np.arange(0, WALL_W, s2), np.arange(-BUILDING_DEPTH, 0, s2))
        parts.append(world_wall(ru.ravel(), np.full(ru.size, WALL_H), rw.ravel()))
        for end in (0.0, WALL_W):
            ev, ew = np.meshgrid(np.arange(0, WALL_H, s2), np.arange(-BUILDING_DEPTH, 0, s2))
            parts.append(world_wall(np.full(ev.size, end), ev.ravel(), ew.ravel()))
    if ground:  # lawn in front of the wall and a return wall at the left corner
        gu, gw = np.meshgrid(np.arange(-1.0, WALL_W + 1.0, 0.05), np.arange(0.1, 5.0, 0.05))
        parts.append(world_wall(gu.ravel(), np.zeros(gu.size), gw.ravel()))
        rv, rw = np.meshgrid(np.arange(0, WALL_H, 0.05), np.arange(0.05, 1.5, 0.05))
        parts.append(world_wall(np.zeros(rv.size), rv.ravel(), rw.ravel()))
    return np.concatenate(parts)


REF_LLA = {"lat": 44.09, "lon": -103.21, "alt": 0.0}  # western SD: ~1.25 deg grid convergence


def write_laz(path: Path, pts_offset: np.ndarray):
    import laspy  # noqa: PLC0415

    header = laspy.LasHeader(point_format=3, version="1.2")
    header.scales = [0.001, 0.001, 0.001]
    header.offsets = [OFFSET_E, OFFSET_N, 0.0]
    las = laspy.LasData(header)
    las.x, las.y, las.z = pts_offset[:, 0] + OFFSET_E, pts_offset[:, 1] + OFFSET_N, pts_offset[:, 2]
    las.write(str(path))


def offset_to_topocentric_fn():
    """Exact offset-frame -> OpenSfM topocentric map (pyproj), for test data."""
    from pyproj import Transformer  # noqa: PLC0415

    to_ll = Transformer.from_crs("EPSG:32613", "EPSG:4326", always_xy=True)
    to_topo = Transformer.from_pipeline(
        "+proj=pipeline +step +proj=unitconvert +xy_in=deg +xy_out=rad "
        "+step +proj=cart +ellps=WGS84 "
        f"+step +proj=topocentric +ellps=WGS84 +lat_0={REF_LLA['lat']} +lon_0={REF_LLA['lon']} +h_0={REF_LLA['alt']}"
    )

    def fn(p):
        p = np.atleast_2d(p)
        lon, lat = to_ll.transform(p[:, 0] + OFFSET_E, p[:, 1] + OFFSET_N)
        x, y, z = to_topo.transform(lon, lat, p[:, 2])
        return np.stack([x, y, z], -1)
    return fn


def write_geo_obj(path: Path, pts_offset: np.ndarray):
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [f"v {x:.4f} {y:.4f} {z:.4f}" for x, y, z in pts_offset]
    path.write_text("mtllib odm_textured_model_geo.mtl\n" + "\n".join(lines) + "\nf 1 2 3\n")


def project_true(R, c, X):
    """Ground-truth projection (independent of facade.cameras): normalized coords."""
    pc = (X - c) @ R.T
    x, y = pc[:, 0] / pc[:, 2], pc[:, 1] / pc[:, 2]
    r2 = x * x + y * y
    k = CAMERA
    radial = 1 + r2 * (k["k1"] + r2 * (k["k2"] + r2 * k["k3"]))
    xd = x * radial + 2 * k["p1"] * x * y + k["p2"] * (r2 + 2 * x * x)
    yd = y * radial + 2 * k["p2"] * x * y + k["p1"] * (r2 + 2 * y * y)
    return k["focal_x"] * xd + k["c_x"], k["focal_y"] * yd + k["c_y"], pc[:, 2]


def write_tracks(root, cams, sparse_offset, frame_mode):
    """OpenSfM tracks.csv (v2) with each sparse point's true detection per photo."""
    lines = ["OPENSFM_TRACKS_VERSION_v2"]
    for name, c, target in cams:
        R = look_at(c, target)
        x, y, z = project_true(R, c, sparse_offset)
        ok = (z > 0) & (np.abs(x) < 0.5) & (np.abs(y) < 0.375)
        for i in np.flatnonzero(ok)[:400]:
            lines.append(f"{name}\t{i}\t{i}\t{x[i]:.6f}\t{y[i]:.6f}\t0.004\t128\t128\t128\t-1\t-1")
    (root / "opensfm" / "tracks.csv").write_text("\n".join(lines) + "\n")


def build_project(root: Path, *, gains=None, with_cloud=True, cloud=None, frame_mode="marker",
                  geo_mesh=False, topo_ply=False, shading=False):
    """frame_mode: "marker" (offset poses + topocentric marker file, like ODM),
    "nomarker" (offset poses, no marker, LAZ only), "topocentric" (poses in
    OpenSfM's local ENU frame, no marker, LAZ only: the Lightning all.zip case)."""
    root = Path(root)
    (root / "opensfm").mkdir(parents=True, exist_ok=True)
    (root / "images").mkdir(exist_ok=True)
    (root / "odm_georeferencing").mkdir(exist_ok=True)
    cams = camera_positions()
    rng = np.random.default_rng(1)
    if gains is None:
        gains = rng.uniform(0.85, 1.15, len(cams))
    shots = {}
    for (name, c, target), g in zip(cams, gains):
        R = look_at(c, target)
        shade = None
        if shading:
            shade = (rng.uniform(-0.35, 0.35), rng.uniform(-0.35, 0.35),
                     (rng.uniform(0.92, 1.08), 1.0, rng.uniform(0.92, 1.08)))
        cv2.imwrite(str(root / "images" / name), render(R, c, g, shade), [cv2.IMWRITE_JPEG_QUALITY, 95])
        rvec, _ = cv2.Rodrigues(R)
        shots[name] = {"camera": "synthetic", "rotation": rvec.ravel().tolist(),
                       "translation": (-R @ c).tolist()}
    dense = point_cloud(**(cloud or {}))
    sparse = dense[rng.choice(len(dense), 3000, replace=False)] + rng.normal(0, 0.01, (3000, 3))
    sparse_offset_frame = sparse.copy()
    if frame_mode == "topocentric":
        to_topo = offset_to_topocentric_fn()
        # local rotation of the frame change, from the exact map
        c0 = world_wall(WALL_W / 2, WALL_H / 2)
        jac = np.stack([(to_topo(c0 + e) - to_topo(c0 - e))[0] / 2 for e in np.eye(3)], -1)
        u, _, vt = np.linalg.svd(jac)
        r_topo = u @ vt
        for (name, c, target) in cams:
            R = look_at(c, target) @ r_topo.T
            ct = to_topo(c)[0]
            rvec, _ = cv2.Rodrigues(R)
            shots[name] = {"camera": "synthetic", "rotation": rvec.ravel().tolist(),
                           "translation": (-R @ ct).tolist()}
        sparse = to_topo(sparse)
        (root / "opensfm" / "reference_lla.json").write_text(json.dumps(
            {"latitude": REF_LLA["lat"], "longitude": REF_LLA["lon"], "altitude": REF_LLA["alt"]}))
    points = {str(i): {"coordinates": p.tolist()} for i, p in enumerate(sparse)}
    write_tracks(root, cams, sparse_offset_frame, frame_mode)
    recon = [{"cameras": {"synthetic": CAMERA}, "shots": shots, "points": points}]
    (root / "opensfm" / "reconstruction.json").write_text(json.dumps(recon))
    if frame_mode == "marker":
        (root / "opensfm" / "reconstruction.topocentric.json").write_text("[]")
    (root / "opensfm" / "image_list.txt").write_text("\n".join(f"images/{n}" for n, *_ in cams))
    (root / "odm_georeferencing" / "coords.txt").write_text(f"WGS84 UTM 13N\n{OFFSET_E:.0f} {OFFSET_N:.0f}\n")
    if with_cloud and frame_mode == "marker":
        (root / "odm_filterpoints").mkdir(exist_ok=True)
        write_ply(root / "odm_filterpoints" / "point_cloud.ply", dense)
    elif with_cloud:
        write_laz(root / "odm_georeferencing" / "odm_georeferenced_model.laz", dense)
    if topo_ply:  # ODM's filterpoints PLY lives in the reconstruction's own frame
        (root / "odm_filterpoints").mkdir(exist_ok=True)
        write_ply(root / "odm_filterpoints" / "point_cloud.ply", offset_to_topocentric_fn()(dense))
    if geo_mesh:
        write_geo_obj(root / "odm_texturing" / "odm_textured_model_geo.obj", dense[::4])
    return {name: g for (name, *_), g in zip(cams, gains)}


def opensfm_to_mesh(p):
    p = np.asarray(p, float)
    e, n = p[0] + OFFSET_E, p[1] + OFFSET_N
    return [e - MESH_ORIGIN["e"], p[2] - MESH_ORIGIN["z"], -(n - MESH_ORIGIN["n"])]


def wall_payload(frame="mesh", swap=False):
    bl, br, tl = world_wall(0, 0), world_wall(WALL_W, 0), world_wall(0, WALL_H)
    if swap:  # user clicked the corners from "inside": mirror image unless flipped
        bl, br, tl = br, bl, world_wall(WALL_W, WALL_H)
    corners = {"bottom_left": bl, "bottom_right": br, "top_left": tl}
    if frame == "mesh":
        corners = {k: opensfm_to_mesh(v) for k, v in corners.items()}
    else:
        corners = {k: list(map(float, v)) for k, v in corners.items()}
    return {"frame": frame, "corners": corners, "mesh_origin": MESH_ORIGIN}


def truth_image(gsd_m, width_px, height_px):
    cols = (np.arange(width_px) + 0.5) * gsd_m
    rows = WALL_H - (np.arange(height_px) + 0.5) * gsd_m
    uu, vv = np.meshgrid(cols, rows)
    return wall_texture(uu, vv)  # BGR float
