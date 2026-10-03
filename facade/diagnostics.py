"""Self-checks against the real dataset, so a bad result explains itself.

* camera_check: OpenSfM's tracks.csv records where each sparse point was
  actually detected in each photo. Projecting those points with this
  worker's camera code and comparing gives a reprojection error in pixels.
  A correct camera model lands within a pixel or two; a misread one (wrong
  convention, lens model, image orientation) lands tens to thousands of
  pixels off. A few alternative interpretations are scored too, so the
  report says which one fits if the default does not.
* overlays: the wall outline and a 1 m grid drawn on the photos that
  contributed most, small JPEGs embedded in the sidecar.
"""
from __future__ import annotations

import base64
import math
from pathlib import Path

import cv2
import numpy as np

from .cameras import Shot
from .images import read_image

MAX_OBS_PER_SHOT = 400


def read_tracks(path: Path, shots: set, max_per_shot: int = MAX_OBS_PER_SHOT):
    """{shot: [(track_id, x_norm, y_norm), ...]} for the requested shots only.

    Handles OpenSfM tracks.csv v0 (no header) and v1/v2 (version header).
    Streams the file: real ones run to millions of lines.
    """
    out = {s: [] for s in shots}
    full = set()
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            if line.startswith("OPENSFM_TRACKS_VERSION"):
                continue
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 5:
                continue
            shot = parts[0]
            lst = out.get(shot)
            if lst is None or shot in full:
                continue
            try:
                lst.append((parts[1], float(parts[3]), float(parts[4])))
            except ValueError:
                continue
            if len(lst) >= max_per_shot:
                full.add(shot)
                if len(full) == len(out):
                    break
    return out


def _errors(shot: Shot, pts: np.ndarray, obs_xy: np.ndarray, R=None, t=None, cam=None, swap=False):
    s = Shot(shot.name, cam or shot.camera, shot.R if R is None else R, shot.t if t is None else t,
             shot.image_path, shot.image_w, shot.image_h)
    px, py, _ = s.project(pts)
    w, h = s.size()
    size = max(w, h)
    ox, oy = (obs_xy[:, 1], obs_xy[:, 0]) if swap else (obs_xy[:, 0], obs_xy[:, 1])
    ox = ox * size + w / 2.0 - 0.5
    oy = oy * size + h / 2.0 - 0.5
    e = np.hypot(px - ox, py - oy)
    return e[np.isfinite(e)]


def camera_check(project, shots: list, max_shots: int = 24) -> dict | None:
    tracks_path = project.root / "opensfm" / "tracks.csv"
    if not tracks_path.is_file() or project.sparse is None or not project.sparse_ids:
        return {"status": "skipped", "reason": "no opensfm/tracks.csv or sparse points in the dataset"}
    rng = np.random.default_rng(0)
    pick = list(shots) if len(shots) <= max_shots else [shots[i] for i in
                                                        rng.choice(len(shots), max_shots, replace=False)]
    obs = read_tracks(tracks_path, {s.name for s in pick})
    index = {pid: i for i, pid in enumerate(project.sparse_ids)}
    variants = {"as_is": [], "no_distortion": [], "rotation_transposed": [], "xy_swapped": []}
    per_shot = []
    for shot in pick:
        rows = [(index[tid], x, y) for tid, x, y in obs.get(shot.name, []) if tid in index]
        if len(rows) < 10:
            continue
        ii = np.array([r[0] for r in rows])
        xy = np.array([[r[1], r[2]] for r in rows])
        pts = project.sparse[ii]
        e = _errors(shot, pts, xy)
        variants["as_is"].append(e)
        per_shot.append({"shot": shot.name, "obs": int(len(e)), "median_px": round(float(np.median(e)), 2)})
        cam0 = type(shot.camera)(shot.camera.id, shot.camera.model, shot.camera.width, shot.camera.height,
                                 {k: (0.0 if k.startswith(("k", "p")) else v)
                                  for k, v in shot.camera.params.items()})
        variants["no_distortion"].append(_errors(shot, pts, xy, cam=cam0))
        variants["rotation_transposed"].append(_errors(shot, pts, xy, R=shot.R.T, t=shot.t))
        variants["xy_swapped"].append(_errors(shot, pts, xy, swap=True))
    if not per_shot:
        return {"status": "skipped", "reason": "no tracks matched the checked photos"}

    def summary(es):
        e = np.concatenate(es) if es else np.zeros(0)
        return {"median_px": round(float(np.median(e)), 2), "p90_px": round(float(np.percentile(e, 90)), 2),
                "observations": int(len(e))} if len(e) else None

    report = {"variants": {k: summary(v) for k, v in variants.items()},
              "shots_checked": len(per_shot), "per_shot": per_shot[:12]}
    med = report["variants"]["as_is"]["median_px"]
    best = min((v["median_px"], k) for k, v in report["variants"].items() if v)
    report["median_px"] = med
    report["best_variant"] = best[1]
    report["status"] = "ok" if med <= 3 else ("marginal" if med <= 10 else "failed")
    return report


def _draw_polyline(img, shot, pts_world, scale, color, thickness=2):
    px, py, d = shot.project(pts_world)
    ok = np.isfinite(px) & (d > 0)
    seg = []
    for x, y, k in zip(px, py, ok):
        if k and abs(x) < 1e6 and abs(y) < 1e6:
            seg.append((x * scale, y * scale))
        elif len(seg) > 1:
            cv2.polylines(img, [np.int32(seg)], False, color, thickness, cv2.LINE_AA)
            seg = []
        else:
            seg = []
    if len(seg) > 1:
        cv2.polylines(img, [np.int32(seg)], False, color, thickness, cv2.LINE_AA)


def overlay(shot: Shot, plane, depth=None, project=None, long_edge: int = 1400, quality: int = 72) -> str:
    """Photo with the wall outline (yellow), 1 m grid (cyan) and, where
    available, tracks.csv detections (red) vs this worker's projections
    (green). Returns a base64 JPEG."""
    img = read_image(shot.image_path, reduce=4)
    w, h = shot.size()
    scale = img.shape[1] / w
    target = long_edge / max(img.shape[:2])
    if target < 1:
        img = cv2.resize(img, None, fx=target, fy=target, interpolation=cv2.INTER_AREA)
        scale *= target
    img = img.copy()
    n = 200

    def line(u0, v0, u1, v1):
        uu, vv = np.linspace(u0, u1, n), np.linspace(v0, v1, n)
        ww = depth.sample(uu, vv) if depth is not None else np.zeros(n)
        return plane.to_world(uu, vv, ww)

    for k in range(0, int(math.floor(plane.width_m)) + 1):
        _draw_polyline(img, shot, line(k, 0, k, plane.height_m), scale, (255, 255, 0), 1)
    for k in range(0, int(math.floor(plane.height_m)) + 1):
        _draw_polyline(img, shot, line(0, k, plane.width_m, k), scale, (255, 255, 0), 1)
    W, H = plane.width_m, plane.height_m
    for a, b in (((0, 0), (W, 0)), ((W, 0), (W, H)), ((W, H), (0, H)), ((0, H), (0, 0))):
        _draw_polyline(img, shot, line(a[0], a[1], b[0], b[1]), scale, (0, 230, 255), 3)
    if project is not None and project.sparse is not None and project.sparse_ids:
        tp = project.root / "opensfm" / "tracks.csv"
        if tp.is_file():
            obs = read_tracks(tp, {shot.name}, 300).get(shot.name, [])
            index = {pid: i for i, pid in enumerate(project.sparse_ids)}
            size = max(w, h)
            for tid, x, y in obs:
                i = index.get(tid)
                if i is None:
                    continue
                ox, oy = x * size + w / 2 - 0.5, y * size + h / 2 - 0.5
                px, py, d = shot.project(project.sparse[i][None])
                cv2.circle(img, (int(ox * scale), int(oy * scale)), 3, (0, 0, 255), -1)
                if np.isfinite(px[0]) and d[0] > 0 and abs(px[0]) < 1e6 and abs(py[0]) < 1e6:
                    cv2.circle(img, (int(px[0] * scale), int(py[0] * scale)), 3, (0, 255, 0), 1)
    cv2.putText(img, shot.name, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, quality])
    return base64.b64encode(buf.tobytes()).decode("ascii") if ok else ""
