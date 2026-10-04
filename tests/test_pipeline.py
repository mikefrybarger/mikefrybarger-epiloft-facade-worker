"""Full pipeline against the synthetic wall: accuracy, occlusion, seams, flips."""
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import synthetic as syn  # noqa: E402
from handler import produce  # noqa: E402

tifffile = pytest.importorskip("tifffile")


@pytest.fixture(scope="module")
def project(tmp_path_factory):
    root = tmp_path_factory.mktemp("odm")
    syn.build_project(root)
    return root


def _run(project, tmp_path, **job):
    job.setdefault("wall", syn.wall_payload())
    sidecar = produce(project, job, tmp_path)
    rgba = tifffile.imread(tmp_path / "facade.tif")
    return sidecar, rgba


def _post_fraction(rgb, alpha):
    """Share of the image showing the magenta post (no wall colour is close)."""
    r, g, b = rgb[..., 0].astype(int), rgb[..., 1].astype(int), rgb[..., 2].astype(int)
    post = (r > 150) & (b > 150) & (g < 90) & (alpha > 0)
    return post.mean()


def _compare(rgba, gsd_m):
    h, w = rgba.shape[:2]
    truth = syn.truth_image(gsd_m, w, h)[..., ::-1]  # BGR -> RGB
    got = rgba[..., :3].astype(np.float64)
    ok = rgba[..., 3] > 0
    m = 8  # ignore the outermost pixels
    ok[:m], ok[-m:], ok[:, :m], ok[:, -m:] = False, False, False, False
    # global brightness differs (gains pull toward 1, truth is gain 1)
    scale = (truth[ok] * got[ok]).sum() / (got[ok] ** 2).sum()
    # Overall brightness must be right, not just fitted away: a facade that
    # comes out dark is a failed facade (see the gain normalisation fix).
    assert 0.85 < scale < 1.15, f"output brightness off by x{1 / scale:.2f}"
    err = got[ok] * scale - truth[ok]
    psnr = 10 * np.log10(255 ** 2 / np.mean(err ** 2))
    shift, _ = cv2.phaseCorrelate(cv2.cvtColor(truth.astype(np.float32), cv2.COLOR_RGB2GRAY),
                                  cv2.cvtColor((got * scale).astype(np.float32), cv2.COLOR_RGB2GRAY))
    return psnr, shift


def test_matches_ground_truth_and_removes_post(project, tmp_path):
    sidecar, rgba = _run(project, tmp_path, gsd_mm=5)
    img = sidecar["image"]
    assert (img["width_px"], img["height_px"]) == (1200, 600)
    assert rgba.shape == (600, 1200, 4)
    assert img["coverage"] > 0.99
    psnr, shift = _compare(rgba, 0.005)
    assert psnr > 27, psnr
    assert abs(shift[0]) < 0.5 and abs(shift[1]) < 0.5, shift
    assert _post_fraction(rgba[..., :3], rgba[..., 3]) < 0.0005
    assert len(sidecar["cameras_used"]) >= 4
    chk = sidecar["diagnostics"]["camera_check"]
    assert chk["status"] == "ok" and chk["median_px"] < 1.0, chk
    assert chk["best_variant"] == "as_is", chk
    assert len(sidecar["diagnostics"]["overlays"]) == 3
    assert sidecar["diagnostics"]["overlays"][0]["jpeg_base64"][:4] == "/9j/"  # JPEG
    assert all("NADIR" not in c["name"] for c in sidecar["cameras_used"])
    assert sidecar["depth"]["surface_points"] > 10_000
    assert abs(sidecar["depth"]["offset_median_m"]) < 0.01
    # deliverables
    assert (tmp_path / "facade_preview.jpg").is_file()
    meta = json.loads((tmp_path / "facade.json").read_text())
    assert meta["kind"] == "epiloft.facade_ortho"
    assert abs(meta["plane"]["width_m"] - syn.WALL_W) < 1e-6


def test_without_occlusion_the_post_smears_in(project, tmp_path):
    """Control for the test above: proves the occlusion check is what removes it."""
    sidecar, rgba = _run(project, tmp_path, gsd_mm=5,
                         options={"use_point_cloud": False})
    assert _post_fraction(rgba[..., :3], rgba[..., 3]) > 0.002
    assert any("occlusion" in w for w in sidecar["warnings"])


def test_native_gsd_tracks_capture_distance(project, tmp_path):
    sidecar, _ = _run(project, tmp_path)
    native = sidecar["image"]["gsd_mm"]
    # ~4.1 m standoff, 960 px focal -> ~4.3 mm/px
    assert 3.8 < native < 4.8, native
    assert sidecar["image"]["gsd_source"] == "native"


def test_inside_out_corners_are_flipped_not_mirrored(project, tmp_path):
    sidecar, rgba = _run(project, tmp_path, gsd_mm=5, wall=syn.wall_payload(swap=True))
    assert sidecar["diagnostics"]["side"]["flipped"] is True, sidecar["diagnostics"]["side"]
    psnr, shift = _compare(rgba, 0.005)
    assert psnr > 27, psnr


def test_opensfm_frame_matches_mesh_frame(project, tmp_path):
    a, rgba_a = _run(project, tmp_path / "a", gsd_mm=10)
    b, rgba_b = _run(project, tmp_path / "b", gsd_mm=10, wall=syn.wall_payload(frame="opensfm"))
    assert rgba_a.shape == rgba_b.shape
    diff = np.abs(rgba_a.astype(int) - rgba_b.astype(int)).mean()
    assert diff < 1.0, diff


def test_exposure_seams_are_levelled(tmp_path):
    root = tmp_path / "odm"
    cams = syn.camera_positions()
    gains = np.where(np.arange(len(cams)) % 2 == 0, 0.75, 1.25)  # harsh alternating exposure
    syn.build_project(root, gains=gains)
    sidecar, rgba = _run(root, tmp_path / "out", gsd_mm=5)
    psnr, _ = _compare(rgba, 0.005)
    assert psnr > 25, psnr
    used = {c["name"]: c["gain_bgr"] for c in sidecar["cameras_used"]}
    applied = {n: g for n, g in zip([c[0] for c in cams], gains)}
    products = [np.mean(used[n]) * applied[n] for n in used]
    assert np.std(products) / np.mean(products) < 0.06, products


def test_rejects_wall_no_photo_faces(project, tmp_path):
    wall = syn.wall_payload(frame="opensfm")
    # rotate the wall 90 degrees so every camera sees it edge-on
    o = np.array(wall["corners"]["bottom_left"])
    wall["corners"]["bottom_right"] = (o + 6.0 * syn.W).tolist()
    with pytest.raises(RuntimeError, match="no photo"):
        produce(project, {"wall": wall, "gsd_mm": 10}, tmp_path)


def test_megapixel_limit(project, tmp_path):
    with pytest.raises(RuntimeError, match="MP limit"):
        produce(project, {"wall": syn.wall_payload(), "gsd_mm": 0.5,
                          "options": {"max_output_megapixels": 10}}, tmp_path)


def _offset_wall(dw):
    """Wall picked dw metres in front of the real surface (e.g. clicked on the eaves line)."""
    wall = syn.wall_payload(frame="opensfm")
    for k, p in wall["corners"].items():
        wall["corners"][k] = (np.array(p) + dw * syn.W).tolist()
    return wall


def test_depth_stage_finds_real_surface_behind_picked_plane(project, tmp_path):
    sidecar, rgba = _run(project, tmp_path, gsd_mm=5, wall=_offset_wall(0.35))
    assert abs(sidecar["depth"]["offset_median_m"] + 0.35) < 0.02
    psnr, shift = _compare(rgba, 0.005)
    assert psnr > 27, psnr
    assert abs(shift[0]) < 0.5 and abs(shift[1]) < 0.5, shift


def test_flat_plane_off_the_surface_ghosts(project, tmp_path):
    """Control: same mis-picked plane without depth gives visibly worse output."""
    _, rgba = _run(project, tmp_path, gsd_mm=5, wall=_offset_wall(0.35),
                   options={"use_point_cloud": False})
    psnr, _ = _compare(rgba, 0.005)
    assert psnr < 24, psnr


def test_deep_zoom_tiles_and_laz_point_cloud(tmp_path):
    pytest.importorskip("pyvips")
    laspy = pytest.importorskip("laspy")
    import zipfile

    root = tmp_path / "odm"
    syn.build_project(root, with_cloud=False)
    pts = syn.point_cloud()
    header = laspy.LasHeader(point_format=3, version="1.2")
    header.scales = [0.001, 0.001, 0.001]
    header.offsets = [syn.OFFSET_E, syn.OFFSET_N, 0.0]
    las = laspy.LasData(header)
    las.x, las.y, las.z = pts[:, 0] + syn.OFFSET_E, pts[:, 1] + syn.OFFSET_N, pts[:, 2]
    las.write(str(root / "odm_georeferencing" / "odm_georeferenced_model.laz"))

    sidecar, rgba = _run(root, tmp_path / "out", gsd_mm=5, make_tiles=True)
    assert sidecar["depth"]["source"].startswith("odm_georeferencing/odm_georeferenced_model.laz")
    assert _post_fraction(rgba[..., :3], rgba[..., 3]) < 0.0005
    with zipfile.ZipFile(tmp_path / "out" / "facade_tiles.zip") as zf:
        names = zf.namelist()
    assert any(n.endswith(".dzi") for n in names) and any(n.endswith(".png") for n in names)
    assert sidecar["files"]["tiles"] == "facade_tiles.zip"


def test_sparse_point_cloud_still_blocks_post(tmp_path):
    """Regression: z-buffer cells sized in image pixels left holes between
    sparse points (seen at 20 MP), letting the post bleed through."""
    root = tmp_path / "odm"
    syn.build_project(root, cloud={"spacing": 0.08, "post_ring": 8, "post_step": 0.08})
    _, rgba = _run(root, tmp_path / "out", gsd_mm=5)
    assert _post_fraction(rgba[..., :3], rgba[..., 3]) < 0.0005


@pytest.mark.parametrize("mode", ["nomarker", "topocentric"])
def test_pose_frame_detected_from_data(tmp_path, mode):
    """The Lightning all.zip case: no reconstruction.topocentric.json marker.

    Topocentric poses are rotated ~1.25 deg and shifted relative to the
    offset frame here, so guessing wrong would wreck the comparison."""
    pytest.importorskip("laspy")
    pytest.importorskip("pyproj")
    root = tmp_path / "odm"
    syn.build_project(root, frame_mode=mode)
    sidecar, rgba = _run(root, tmp_path / "out", gsd_mm=5)
    frame = sidecar["wall_to_world"]["pose_frame"]
    expected = "offset" if mode == "nomarker" else "topocentric"
    assert frame["method"].startswith(expected), frame
    assert frame["fit_m"] < 0.05, frame
    psnr, shift = _compare(rgba, 0.005)
    assert psnr > 27, psnr
    assert abs(shift[0]) < 0.5 and abs(shift[1]) < 0.5, shift
    assert _post_fraction(rgba[..., :3], rgba[..., 3]) < 0.0005
    assert not any("frame" in w for w in sidecar["warnings"]), sidecar["warnings"]


def test_topocentric_checked_against_textured_mesh(tmp_path):
    """No LAZ, but ODM's georeferenced textured mesh is in the archive."""
    pytest.importorskip("pyproj")
    root = tmp_path / "odm"
    syn.build_project(root, frame_mode="topocentric", with_cloud=False, geo_mesh=True)
    sidecar, rgba = _run(root, tmp_path / "out", gsd_mm=5)
    frame = sidecar["wall_to_world"]["pose_frame"]
    assert frame["method"].startswith("topocentric") and frame["reference"].endswith(".obj"), frame
    psnr, _ = _compare(rgba, 0.005)
    assert psnr > 27, psnr


def test_unverifiable_frame_is_explained(tmp_path):
    """Nothing to check against and the guess is wrong: the error says why."""
    root = tmp_path / "odm"
    syn.build_project(root, frame_mode="topocentric", with_cloud=False)
    with pytest.raises(RuntimeError, match="could not verify the camera frame"):
        _run(root, tmp_path / "out", gsd_mm=10)


def test_topocentric_ply_moves_with_the_poses(tmp_path):
    """PLY in the poses' original frame + textured mesh as the reference: the
    PLY must get the same frame change, or occlusion and depth miss the wall."""
    pytest.importorskip("pyproj")
    root = tmp_path / "odm"
    syn.build_project(root, frame_mode="topocentric", with_cloud=False, geo_mesh=True, topo_ply=True)
    sidecar, rgba = _run(root, tmp_path / "out", gsd_mm=5)
    assert sidecar["depth"]["source"].startswith("odm_filterpoints/point_cloud.ply")
    assert abs(sidecar["depth"]["offset_median_m"]) < 0.02, sidecar["depth"]
    assert _post_fraction(rgba[..., :3], rgba[..., 3]) < 0.0005
    psnr, _ = _compare(rgba, 0.005)
    assert psnr > 27, psnr


def _shift_poses(root, delta):
    """Simulate a datum / georeferencing error: every pose and sparse point off by delta."""
    p = root / "opensfm" / "reconstruction.json"
    recon = json.loads(p.read_text())
    for shot in recon[0]["shots"].values():
        R = cv2.Rodrigues(np.array(shot["rotation"], float))[0]
        c = -R.T @ np.array(shot["translation"]) + delta
        shot["translation"] = (-R @ c).tolist()
    for pt in recon[0]["points"].values():
        pt["coordinates"] = (np.array(pt["coordinates"]) + delta).tolist()
    p.write_text(json.dumps(recon))


def test_local_snap_corrects_pose_offset(tmp_path):
    """Ground and a return wall make all three directions observable; a lone
    flat wall can only pin the direction perpendicular to it."""
    pytest.importorskip("scipy")
    delta = np.array([0.12, -0.08, 0.05])
    root = tmp_path / "odm"
    syn.build_project(root, frame_mode="marker", cloud={"ground": True})
    _shift_poses(root, delta)
    sidecar, rgba = _run(root, tmp_path / "out", gsd_mm=5)
    frame = sidecar["wall_to_world"]["pose_frame"]
    assert frame["constrained_axes"] == 3, frame
    assert np.allclose(frame["local_snap_m"], -delta, atol=0.01), frame
    assert frame["local_fit_m"] < 0.02, frame
    psnr, shift = _compare(rgba, 0.005)
    assert psnr > 27, psnr
    assert abs(shift[0]) < 0.5 and abs(shift[1]) < 0.5, shift


def test_local_snap_never_invents_unconstrained_motion(tmp_path):
    """Wall + vertical post, no ground: the wall pins through-wall motion, the
    post's curved side pins along-wall motion, nothing pins vertical."""
    pytest.importorskip("scipy")
    delta = np.array([0.12, -0.08, 0.05])
    root = tmp_path / "odm"
    syn.build_project(root, frame_mode="marker")
    _shift_poses(root, delta)
    sidecar, _ = _run(root, tmp_path / "out", gsd_mm=5)
    frame = sidecar["wall_to_world"]["pose_frame"]
    snap = np.array(frame["local_snap_m"])
    assert frame["constrained_axes"] == 2, frame
    assert abs(snap @ syn.W + delta @ syn.W) < 0.01, frame   # through-wall error removed
    assert abs(snap @ syn.U + delta @ syn.U) < 0.02, frame   # along-wall, via the post
    assert abs(snap @ syn.V) < 0.005, frame                  # vertical: nothing invented


def test_many_photos_do_not_drift_dark():
    """Regression: on a real 80-photo job every gain sat on the 0.5 floor and
    the facade came out dark. The pairwise terms only fix relative gains;
    systematic differences between overlapping photos (shadows moving during
    the flight, glass, sheen) make shrinking every gain look cheaper. The old
    solve drops to ~0.74 on this case and keeps falling as photos disagree more."""
    from facade.selection import SelectionConfig, solve_gains

    rng = np.random.default_rng(0)
    n, cells = 80, 4000
    true_gain = rng.uniform(0.8, 1.2, n)
    scene = rng.uniform(40, 220, (cells, 3))
    valid = np.zeros((n, cells), bool)
    for i in range(n):  # heavy overlap along a long wall
        start = int(i * cells / n)
        valid[i, max(0, start - 750): start + 750] = True
    smooth = np.repeat(rng.normal(0, 0.2, (n, cells // 100 + 1)), 100, axis=1)[:, :cells]
    samples = scene[None] * true_gain[:, None, None] * (1 + smooth)[:, :, None]
    gains = solve_gains(samples, valid, SelectionConfig())
    corrected = gains.mean(axis=1) * true_gain
    assert 0.95 < np.median(gains) < 1.05, np.median(gains)
    assert (gains > 0.5 + 1e-6).all() and (gains < 2.0 - 1e-6).all()
    assert np.std(corrected) / np.mean(corrected) < 0.08


def test_holey_point_cloud_still_lands_on_the_wall(tmp_path):
    """Regression for the real Ascend Plaza job: gaps in the cloud plus a fine
    depth grid made the old hole filler invent metre-scale depths, and the
    facade came out as smeared roofs and parking lot."""
    root = tmp_path / "odm"
    syn.build_project(root, cloud={"spacing": 0.015, "dropout": 0.4})
    sidecar, rgba = _run(root, tmp_path / "out", gsd_mm=5)
    d = sidecar["depth"]
    assert d["coverage_before_fill"] < 0.95, d
    assert -0.08 < d["offset_min_m"] and d["offset_max_m"] < 0.08, d
    psnr, shift = _compare(rgba, 0.005)
    assert psnr > 27, psnr
    assert abs(shift[0]) < 0.5 and abs(shift[1]) < 0.5, shift


def test_camera_check_catches_a_misread_camera(project, tmp_path, monkeypatch):
    """If the camera code were wrong (here: rotation read transposed), the
    check must say so and name the interpretation that does fit."""
    import facade.cameras as cams

    real = cams.rodrigues
    monkeypatch.setattr(cams, "rodrigues", lambda r: real(r).T)
    with pytest.raises(RuntimeError, match=r"Camera check failed.*best fit: rotation_transposed"):
        _run(project, tmp_path, gsd_mm=20)


M4E = {"projection_type": "brown", "focal_x": 0.7052558642285107, "focal_y": 0.7052558642285107,
       "c_x": 0.004969223516840049, "c_y": -0.00470324524625981, "k1": -0.10341031555259611,
       "k2": -0.01531662859893411, "p1": 2.6876150908751606e-06, "p2": -0.00020501113013549115,
       "k3": -0.005277757045775089}


def test_real_m4e_lens_and_photos_looking_along_the_wall(tmp_path, monkeypatch):
    """Regression for the Ascend Plaza job: with the real DJI M4E calibration,
    rays past 53 deg off-axis fold back into the photo (63.5 deg lands on the
    image centre). Photos parked in front of the wall but aimed along it were
    "seeing" wall that was outside their frame, and the facade was painted
    from pavement and roofs."""
    monkeypatch.setattr(syn, "CAMERA", dict(syn.CAMERA, **M4E))
    base = syn.camera_positions

    def with_decoys():
        cams = base()
        for i, uu in enumerate((0.5, 2.5, 4.5)):
            c = syn.world_wall(uu, 1.6, 3.0)
            target = c + 10.0 * syn.U - 2.5 * syn.W     # mostly along the wall
            cams.append((f"DECOY_{i}.JPG", c, target))
        return cams
    monkeypatch.setattr(syn, "camera_positions", with_decoys)
    root = tmp_path / "odm"
    syn.build_project(root)
    sidecar, rgba = _run(root, tmp_path / "out", gsd_mm=5)
    used = [c["name"] for c in sidecar["cameras_used"]]
    assert not any(n.startswith("DECOY") and c["share"] > 0.02
                   for n, c in zip(used, sidecar["cameras_used"])), sidecar["cameras_used"]
    psnr, shift = _compare(rgba, 0.005)
    assert psnr > 26, psnr
    assert abs(shift[0]) < 0.5 and abs(shift[1]) < 0.5, shift


def _with_back_side(monkeypatch):
    """Ascend Plaza layout: the facade is the front of a building, and the
    orbit took more photos of the back. Back photos aim at the back wall,
    which also points them "toward" the facade, straight through the building."""
    base = syn.camera_positions

    def cams():
        out = base()
        for i, uu in enumerate(np.linspace(0.3, syn.WALL_W - 0.3, 24)):
            c = syn.world_wall(uu, 6.0, -syn.BUILDING_DEPTH - 12.0)
            target = syn.world_wall(uu, 1.5, -syn.BUILDING_DEPTH)
            out.append((f"BACK_{i:02d}.JPG", c, target))
        return out
    monkeypatch.setattr(syn, "camera_positions", cams)


def test_back_of_building_photos_never_paint_the_facade(tmp_path, monkeypatch):
    from facade.geometry import WallPlane

    _with_back_side(monkeypatch)
    root = tmp_path / "odm"
    syn.build_project(root, cloud={"building": True})
    # the old rule (count photos pointing at the plane) picks the back here
    plane = WallPlane.from_corners(syn.world_wall(0, 0), syn.world_wall(syn.WALL_W, 0),
                                   syn.world_wall(0, syn.WALL_H))
    cams = syn.camera_positions()
    centers = np.array([c for _, c, _ in cams])
    axes = np.array([syn.look_at(c, t)[2] for _, c, t in cams])
    assert plane.face_cameras(centers, axes), "scenario should fool the old vote"

    sidecar, rgba = _run(root, tmp_path / "out", gsd_mm=5)
    side = sidecar["diagnostics"]["side"]
    assert side["method"] == "visibility" and side["flipped"] is False, side
    assert side["visible_pairs"]["as_picked"] > 5 * max(1, side["visible_pairs"]["reverse"]), side
    assert not any(c["name"].startswith("BACK") for c in sidecar["cameras_used"])
    psnr, shift = _compare(rgba, 0.005)
    assert psnr > 27, psnr


def test_view_from_decides_the_side(tmp_path, monkeypatch):
    _with_back_side(monkeypatch)
    root = tmp_path / "odm"
    syn.build_project(root, cloud={"building": True})
    wall = syn.wall_payload(swap=True)               # picked "inside out"
    wall["view_from"] = syn.opensfm_to_mesh(syn.world_wall(3.0, 1.7, 6.0))   # stood in front
    sidecar, rgba = _run(root, tmp_path / "out", gsd_mm=5, wall=wall)
    side = sidecar["diagnostics"]["side"]
    assert side == {"method": "view_from", "flipped": True}, side
    assert not any(c["name"].startswith("BACK") for c in sidecar["cameras_used"])
    psnr, _ = _compare(rgba, 0.005)
    assert psnr > 27, psnr


def _shift_cloud(root, delta):
    """Move the dense cloud (not the poses): the cloud/pose disagreement seen
    at the Ascend Plaza wall (~9 cm)."""
    from facade.pointcloud import read_ply_xyz
    p = root / "odm_filterpoints" / "point_cloud.ply"
    syn.write_ply(p, read_ply_xyz(p) + delta)


def test_photo_consistency_removes_cloud_pose_mismatch(tmp_path):
    root = tmp_path / "odm"
    syn.build_project(root)
    _shift_cloud(root, 0.08 * syn.W)              # cloud 8 cm in front of where the photos say
    opts = {"local_snap": False}
    off, rgba_off = _run(root, tmp_path / "off", gsd_mm=5, options={**opts, "refine_depth": False})
    on, rgba_on = _run(root, tmp_path / "on", gsd_mm=5, options=opts)
    psnr_off, _ = _compare(rgba_off, 0.005)
    psnr_on, shift_on = _compare(rgba_on, 0.005)
    ref = on["diagnostics"]["refine"]
    assert abs(ref["global_offset_m"] + 0.08) < 0.015, ref      # found the 8 cm
    assert psnr_on > psnr_off + 3, (psnr_off, psnr_on)          # and it shows
    assert psnr_on > 30, psnr_on
    assert abs(shift_on[0]) < 0.5 and abs(shift_on[1]) < 0.5


def test_ground_in_front_does_not_drag_the_base_onto_the_pavement(tmp_path):
    root = tmp_path / "odm"
    syn.build_project(root, cloud={"ground": True})
    sidecar, rgba = _run(root, tmp_path / "out", gsd_mm=5)
    assert sidecar["depth"]["ground_points_dropped"] > 1000
    bottom = rgba[-40:]                                          # lowest 20 cm of the wall
    h, w = rgba.shape[:2]
    truth = syn.truth_image(0.005, w, h)[-40:, :, ::-1]
    ok = bottom[..., 3] > 0
    err = np.abs(bottom[..., :3].astype(float) - truth)[ok].mean()
    assert err < 12, err


def test_sky_above_the_parapet_is_transparent(tmp_path):
    root = tmp_path / "odm"
    syn.build_project(root)
    wall = syn.wall_payload(frame="opensfm")
    wall["corners"]["top_left"] = list(map(float, syn.world_wall(0, syn.WALL_H + 1.0)))  # 1 m too tall
    sidecar, rgba = _run(root, tmp_path / "out", gsd_mm=10, wall=wall)
    top = rgba[:80, :, 3]                                        # top 0.8 m: nothing there
    body = rgba[150:, :, 3]
    assert (top == 0).mean() > 0.95, (top == 0).mean()
    assert (body > 0).mean() > 0.97


def _blotchiness(rgba, gsd_m):
    """Low-frequency brightness error: what reads as blotches on plain wall."""
    h, w = rgba.shape[:2]
    truth = syn.truth_image(gsd_m, w, h)[..., ::-1].mean(-1)
    got = rgba[..., :3].astype(np.float64).mean(-1)
    ok = rgba[..., 3] > 0
    scale = (truth[ok] * got[ok]).sum() / (got[ok] ** 2).sum()
    ratio = np.where(ok, got * scale / np.maximum(truth, 1), 1.0)
    k = int(0.5 / gsd_m) | 1                          # 0.5 m blur: blotch scale, not texture
    lf = cv2.GaussianBlur(ratio.astype(np.float32), (k, k), 0)
    m = k // 2
    return float(np.std(lf[m:-m, m:-m]))


def test_local_gains_remove_blotches(tmp_path):
    root = tmp_path / "odm"
    syn.build_project(root, shading=True)
    _, off = _run(root, tmp_path / "off", gsd_mm=10, options={"local_gains": False})
    _, on = _run(root, tmp_path / "on", gsd_mm=10)
    b_off, b_on = _blotchiness(off, 0.01), _blotchiness(on, 0.01)
    assert b_on < 0.6 * b_off, (b_off, b_on)
    assert b_on < 0.04, b_on


def test_stepped_storefront_with_glass_does_not_wobble(tmp_path, monkeypatch):
    """Real storefront shape: glass line set back under a band, sparse shop
    interior through the glass, reflections that differ per photo. The single
    plane + protrusions model called half of this "structure" and per-patch
    corrections jittered on the glass (melting storefront bottoms)."""
    monkeypatch.setitem(syn.STOREFRONT, "enabled", True)
    root = tmp_path / "odm"
    syn.build_project(root)
    sidecar, rgba = _run(root, tmp_path / "out", gsd_mm=5)
    d = sidecar["depth"]
    offsets = sorted(l["offset_m"] for l in d["layers"])
    assert any(abs(o) < 0.03 for o in offsets) and any(abs(o - syn.STORE_W) < 0.03 for o in offsets), d
    assert d["structure_fraction"] < 0.15, d
    ref = sidecar["diagnostics"]["refine"]
    assert ref["segment_offset_spread_m"] < 0.02, ref
    psnr, shift = _compare(rgba, 0.005)
    assert psnr > 27, psnr
    assert abs(shift[0]) < 0.5 and abs(shift[1]) < 0.5, shift


def test_sills_and_frames_do_not_punch_holes(tmp_path):
    """Regression: sills, frames and sign undersides a few cm off the surface
    were treated as obstacles and left white "snow" holes along every sill."""
    root = tmp_path / "odm"
    syn.build_project(root, cloud={"details": True})
    sidecar, rgba = _run(root, tmp_path / "out", gsd_mm=5)
    assert sidecar["diagnostics"]["facade_detail_points"] > 10_000
    assert (rgba[..., 3] > 0).mean() > 0.995, (rgba[..., 3] > 0).mean()
    psnr, _ = _compare(rgba, 0.005)
    assert psnr > 27, psnr
    assert _post_fraction(rgba[..., :3], rgba[..., 3]) < 0.0005     # real obstacles still removed
