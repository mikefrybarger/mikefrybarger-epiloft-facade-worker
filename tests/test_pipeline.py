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
    assert any("flipped" in w for w in sidecar["warnings"])
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
    assert sidecar["depth"]["source"].endswith(".laz")
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
    assert sidecar["depth"]["source"].endswith("point_cloud.ply")
    assert abs(sidecar["depth"]["offset_median_m"]) < 0.02, sidecar["depth"]
    assert _post_fraction(rgba[..., :3], rgba[..., 3]) < 0.0005
    psnr, _ = _compare(rgba, 0.005)
    assert psnr > 27, psnr
