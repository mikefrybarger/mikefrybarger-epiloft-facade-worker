"""Streaming geometry reads: chunked results must match whole-file results."""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import synthetic as syn  # noqa: E402
from facade.cloud import CloudSource, _Capped  # noqa: E402
from facade.geometry import WallPlane  # noqa: E402


def _plane():
    return WallPlane.from_corners(syn.world_wall(0, 0), syn.world_wall(syn.WALL_W, 0), syn.world_wall(0, syn.WALL_H))


def _sources(tmp_path, pts):
    pytest.importorskip("laspy")
    syn.write_ply(tmp_path / "c.ply", pts)
    syn.write_laz(tmp_path / "c.laz", pts)
    syn.write_geo_obj(tmp_path / "m" / "c.obj", pts)
    return [CloudSource("ply", tmp_path / "c.ply", "c.ply", chunk_points=997),
            CloudSource("laz", tmp_path / "c.laz", "c.laz", syn.OFFSET_E, syn.OFFSET_N, chunk_points=997),
            CloudSource("mesh", tmp_path / "m" / "c.obj", "c.obj", chunk_points=997)]


def test_chunked_reads_match_whole_file(tmp_path):
    pts = syn.point_cloud()
    for src in _sources(tmp_path, pts):
        got = np.concatenate(list(src.chunks()))
        assert got.shape == pts.shape, src.kind
        tol = 2e-3 if src.kind == "laz" else 1e-3  # LAZ stores mm, PLY float32, OBJ 4 dp
        assert np.abs(got - pts).max() < tol, src.kind


def test_wall_region_keeps_full_surface_and_caps_occluders(tmp_path):
    pts = syn.point_cloud()
    plane = _plane()
    for src in _sources(tmp_path, pts):
        surface, between, stats = src.wall_region(plane, depth_front_m=0.6, depth_back_m=0.5,
                                                  reach_m=5.0, max_occluders=500)
        uvw = plane.to_wall(pts)
        on_wall = ((uvw[:, 0] >= -0.5) & (uvw[:, 0] <= syn.WALL_W + 0.5) & (uvw[:, 1] >= -0.5)
                   & (uvw[:, 1] <= syn.WALL_H + 0.5) & (np.abs(uvw[:, 2]) <= 0.6))
        assert abs(len(surface) - on_wall.sum()) <= 2, src.kind   # full density on the wall
        assert 0 < len(between) <= 500 and stats["occluders_sampled"], src.kind


def test_sample_is_uniform_and_capped(tmp_path):
    pts = syn.point_cloud()
    src = _sources(tmp_path, pts)[1]
    s = src.sample(5000)
    assert len(s) <= 5000 and len(s) > 4000
    # spread over the whole wall, not one chunk's worth
    uvw = _plane().to_wall(s)
    assert uvw[:, 0].min() < 0.5 and uvw[:, 0].max() > syn.WALL_W - 0.5


def test_capped_thinning_stays_uniform():
    rng = np.random.default_rng(0)
    c = _Capped(1000, rng)
    for i in range(50):
        c.add(np.full((500, 3), float(i)))
    out = c.result()
    assert len(out) <= 1000 and c.halvings > 0
    counts = np.bincount(out[:, 0].astype(int), minlength=50)
    assert counts.min() > 0 and counts.max() < 4 * counts.mean()
