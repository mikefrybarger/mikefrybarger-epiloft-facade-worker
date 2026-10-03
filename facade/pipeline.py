"""The facade ortho pipeline: plane in, full-resolution blended wall image out.

    1  load cameras (OpenSfM)            facade/cameras.py
    2  output grid on the wall plane     facade/geometry.py
    3  true surface depth                facade/depth.py
    4  per-pixel photo choice + occlusion facade/selection.py, visibility.py
    5  exposure gains + multiband blend  facade/selection.py, blend.py
    6  TIFF, sidecar JSON, preview, tiles facade/outputs.py
"""
from __future__ import annotations

import math
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import cv2
import numpy as np

from .blend import multiband_blend
from .depth import build_depth_map, flat_depth
from .georef import local_snap
from .geometry import OrthoGrid, WallPlane
from .images import ImageCache, read_image
from .selection import SelectionConfig, mode_filter, prefilter_shots, score_views, solve_gains
from .visibility import VisibilityConfig, ZBuffer, auto_downscale


@dataclass
class FacadeOptions:
    gsd_mm: float | None = None          # None = native resolution of the best photos
    depth_front_m: float = 0.6           # surface search in front of the plane (trim, sills)
    depth_back_m: float = 0.5            # and behind it (recessed windows, doors)
    depth_cell_mm: float | None = None   # None = from point density
    coarse_long_edge: int = 400          # coarse selection grid cells on the long edge
    tile_px: int = 1024
    blend_levels: int = 5
    max_output_megapixels: float = 600.0
    max_occluder_points: int = 8_000_000
    image_cache_gb: float = 8.0
    use_point_cloud: bool = True
    selection: SelectionConfig = field(default_factory=SelectionConfig)
    visibility: VisibilityConfig = field(default_factory=VisibilityConfig)

    @classmethod
    def from_payload(cls, data: dict | None):
        data = dict(data or {})
        sel = SelectionConfig(**{k: data.pop(k) for k in list(data) if k in SelectionConfig.__dataclass_fields__})
        vis_keys = {"zbuffer_downscale": "downscale", "zbuffer_close_px": "close_px",
                    "zbuffer_cells_per_spacing": "cells_per_spacing",
                    "occlusion_abs_tol_m": "abs_tol_m", "occlusion_rel_tol": "rel_tol"}
        vis = VisibilityConfig(**{vis_keys[k]: data.pop(k) for k in list(data) if k in vis_keys})
        unknown = [k for k in data if k not in cls.__dataclass_fields__]
        if unknown:
            raise ValueError(f"unknown option(s): {', '.join(sorted(unknown))}")
        return cls(selection=sel, visibility=vis, **data)


def _log(progress, msg):
    print(msg, flush=True)
    if progress:
        progress(msg)


def run_facade(project, plane: WallPlane, opts: FacadeOptions, workdir: Path, progress=None) -> dict:
    t0 = time.time()
    timings = {}
    warnings = list(plane.notes)
    if project.frame_report:
        warnings.extend(project.frame_report.get("warnings", []))
    shots = list(project.shots.values())
    if not shots:
        raise RuntimeError("no photos with solved poses were found in the dataset")

    # --- orient the plane toward the photographer ---------------------------
    centers = np.array([s.center for s in shots])
    axes = np.array([s.optical_axis for s in shots])
    if plane.face_cameras(centers, axes):
        warnings.append("wall normal was flipped to face the cameras; image reads left to right from outside")

    # --- candidate photos (decides how far out the cloud needs reading) ------
    sel = opts.selection
    candidates = prefilter_shots(shots, plane, sel)
    if not candidates:
        hint = ""
        if project.frame_report and project.frame_report.get("warnings"):
            hint = " Note: " + "; ".join(project.frame_report["warnings"])
        raise RuntimeError("no photos look at this wall from in front of it; check the wall corners "
                           "or fly oblique passes facing this side." + hint)
    reach = float(max(((s.center - plane.origin) @ plane.w) for s in candidates)) + 1.0

    # --- stage 3: surface depth ---------------------------------------------
    phase = time.time()
    points_wall, occluders = None, np.zeros((0, 3))
    point_spacing = 0.05  # fallback when the cloud does not cover the wall
    pc_source = project.points_source
    if opts.use_point_cloud and project.cloud is not None:
        _log(progress, f"Reading geometry near the wall from {pc_source}")
        surface, between, cloud_stats = project.cloud.wall_region(
            plane, depth_front_m=opts.depth_front_m, depth_back_m=opts.depth_back_m,
            reach_m=reach, max_occluders=opts.max_occluder_points)
        if cloud_stats["surface_capped"]:
            warnings.append("wall surface had more points than the cap; depth used a uniform sample")
        region = np.concatenate([surface, between])
        del between
        if project.frame_report is not None:
            # fine snap against full-density geometry, before occluders are capped
            local = local_snap(project.shots, project.sparse, region, plane)
            warnings.extend(local.pop("warnings"))
            project.frame_report.update(local)
            _log(progress, f"Local alignment at the wall: fit {local['local_fit_m']} m, "
                           f"snap {local['local_snap_m']}")
        occluders = region
        if len(occluders) > opts.max_occluder_points:
            rng = np.random.default_rng(0)
            occluders = occluders[rng.choice(len(occluders), opts.max_occluder_points, replace=False)]
        del region
        points_wall = plane.to_wall(surface) if len(surface) else None
    if points_wall is not None:
        near = ((points_wall[:, 0] >= 0) & (points_wall[:, 0] < plane.width_m)
                & (points_wall[:, 1] >= 0) & (points_wall[:, 1] < plane.height_m)
                & (points_wall[:, 2] <= opts.depth_front_m) & (points_wall[:, 2] >= -opts.depth_back_m))
        n_surface = int(near.sum())
        if n_surface:
            point_spacing = math.sqrt(plane.width_m * plane.height_m / n_surface)
        if opts.depth_cell_mm:
            cell = opts.depth_cell_mm / 1000.0
        elif n_surface:
            cell = float(np.clip(1.5 * point_spacing, 0.01, 0.10))
        else:
            cell = 0.05
        depth = build_depth_map(points_wall, plane.width_m, plane.height_m, cell_m=cell,
                                depth_front_m=opts.depth_front_m, depth_back_m=opts.depth_back_m,
                                source=pc_source)
    elif not opts.use_point_cloud:
        depth = flat_depth(plane.height_m, "point cloud disabled")
    elif project.cloud is None:
        depth = flat_depth(plane.height_m, "no point cloud in dataset")
    else:
        depth = flat_depth(plane.height_m, "point cloud has no points on this wall")
    if depth.point_count == 0:
        warnings.append(f"surface depth: {depth.source}; trim and recesses may show slight parallax")
    timings["depth_seconds"] = round(time.time() - phase, 1)
    _log(progress, f"Surface depth: {depth.source}")
    if len(occluders) == 0:
        warnings.append("no point cloud for occlusion checks; objects in front of the wall may smear onto it")

    # --- stage 4: coarse selection pass ---------------------------------------
    phase = time.time()
    _log(progress, f"Scoring {len(candidates)} candidate photos")
    coarse_cell = max(max(plane.width_m, plane.height_m) / opts.coarse_long_edge, 0.02)
    cgrid = OrthoGrid.for_plane(plane, coarse_cell)
    cu, cv = cgrid.uv(0, cgrid.height_px, 0, cgrid.width_px)
    cpts = plane.to_world(cu, cv, depth.sample(cu, cv))
    # Pass A: geometry-only scores for every candidate (cheap).
    n_all = len(candidates)
    raw_scores = np.full((n_all, cgrid.height_px, cgrid.width_px), -np.inf, dtype=np.float32)
    projections = []
    for k, shot in enumerate(candidates):
        px, py, d = shot.project(cpts)
        sc, g = score_views(shot, cpts, plane.w, px, py, d, sel)
        raw_scores[k] = sc
        projections.append((px, py, d, g))

    # Pass B: occlusion only for photos that could win somewhere. Each round
    # takes the top few photos at every still-uncovered cell; cells whose best
    # photos turn out to be blocked get the next ones in the following round.
    tried = np.zeros(n_all, dtype=bool)
    shortlist = []
    vis_scores, vis_gsds, zbuffers = {}, {}, {}
    uncovered = np.isfinite(raw_scores).any(axis=0)
    for _round in range(4):
        if not uncovered.any():
            break
        masked = np.where(tried[:, None, None], -np.inf, raw_scores)
        masked = np.where(uncovered[None], masked, -np.inf)
        top = min(sel.top_per_cell, n_all)
        best = np.argpartition(-masked.reshape(n_all, -1), top - 1, axis=0)[:top]
        best_ok = np.take_along_axis(masked.reshape(n_all, -1), best, axis=0) > -np.inf
        picks = np.unique(best[best_ok])
        picks = [int(k) for k in picks if not tried[k]]
        if not picks:
            break
        for k in picks:
            tried[k] = True
            shot = candidates[k]
            px, py, d, g = projections[k]
            sc = raw_scores[k]
            dist = float(np.nanmedian(d[np.isfinite(sc)]))
            zb = ZBuffer(shot, occluders, opts.visibility,
                         downscale=opts.visibility.downscale
                         or auto_downscale(shot, dist, point_spacing, opts.visibility))
            sc = np.where(zb.visible(px, py, d), sc, -np.inf)
            if np.isfinite(sc).any():
                shortlist.append(k)
                vis_scores[k], vis_gsds[k], zbuffers[shot.name] = sc, g, zb
        covered_now = np.zeros_like(uncovered)
        for k in shortlist:
            covered_now |= np.isfinite(vis_scores[k])
        uncovered = np.isfinite(raw_scores).any(axis=0) & ~covered_now
    del raw_scores

    n = len(shortlist)
    if n == 0:
        raise RuntimeError("no photo sees this wall unobstructed; check the wall corners and the capture")
    candidates_checked = int(tried.sum())
    _log(progress, f"Occlusion-checked {candidates_checked} of {n_all} photos; {n} see the wall")
    scores = np.stack([vis_scores[k] for k in shortlist])
    gsds = np.stack([vis_gsds[k] for k in shortlist])
    samples = np.zeros((n, cgrid.height_px, cgrid.width_px, 3), dtype=np.float32)
    candidates = [candidates[k] for k in shortlist]
    for j, k in enumerate(shortlist):
        shot = candidates[j]
        px, py, _, _ = projections[k]
        small = read_image(shot.image_path, reduce=8)
        sw, sh = shot.size()
        fx, fy = small.shape[1] / sw, small.shape[0] / sh
        ok = np.isfinite(scores[j])
        mx = np.where(ok, (px + 0.5) * fx - 0.5, -1.0).astype(np.float32)
        my = np.where(ok, (py + 0.5) * fy - 0.5, -1.0).astype(np.float32)
        samples[j] = cv2.remap(small, mx, my, cv2.INTER_LINEAR,
                               borderMode=cv2.BORDER_REPLICATE).astype(np.float32)
    del projections

    valid = np.isfinite(scores)
    covered = valid.any(axis=0)
    if not covered.any():
        raise RuntimeError("no photo sees this wall unobstructed; check the wall corners and the capture")

    # keep the photos that actually win somewhere, best first
    raw = np.where(covered, np.where(valid, scores, -np.inf).argmax(axis=0), -1)
    wins = np.array([(raw == k).sum() for k in range(n)])
    order = [k for k in np.argsort(-wins) if wins[k] >= sel.min_cells][: sel.max_cameras]
    if not order:
        order = [int(np.argmax(wins))]
    order = sorted(order)
    kept = [candidates[k] for k in order]
    scores, gsds, samples, valid = scores[order], gsds[order], samples[order], valid[order]
    keep_names = {s.name for s in kept}
    zbuffers = {name: zb for name, zb in zbuffers.items() if name in keep_names}

    # smooth the winner map so seams follow regions, not speckle
    sm = np.empty_like(scores)
    for k in range(len(kept)):
        filled = np.where(valid[k], scores[k], 0.0).astype(np.float32)
        wgt = cv2.GaussianBlur(valid[k].astype(np.float32), (0, 0), 1.5)
        blur = cv2.GaussianBlur(filled, (0, 0), 1.5) / np.maximum(wgt, 1e-6)
        sm[k] = np.where(valid[k], blur, -np.inf)
    coarse_labels = np.where(valid.any(axis=0), sm.argmax(axis=0), -1)
    coarse_labels = mode_filter(coarse_labels, valid, sel.smooth_passes)
    coverage_coarse = float((coarse_labels >= 0).mean())
    timings["selection_seconds"] = round(time.time() - phase, 1)

    # --- output GSD -----------------------------------------------------------
    win_gsd = np.concatenate([gsds[k][coarse_labels == k] for k in range(len(kept))])
    native_m = float(np.median(win_gsd)) if win_gsd.size else 0.005
    if opts.gsd_mm:
        gsd_m = opts.gsd_mm / 1000.0
        gsd_source = "requested"
        if gsd_m < 0.7 * native_m:
            warnings.append(
                f"requested {opts.gsd_mm} mm/px is finer than the photos resolve "
                f"(~{native_m * 1000:.2f} mm/px median); the extra pixels add file size, not detail"
            )
    else:
        gsd_m = max(round(native_m * 10000) / 10000, 0.0005)
        gsd_source = "native"
    grid = OrthoGrid.for_plane(plane, gsd_m)
    mp = grid.width_px * grid.height_px / 1e6
    if mp > opts.max_output_megapixels:
        need = math.sqrt(mp / opts.max_output_megapixels) * gsd_m * 1000
        raise RuntimeError(
            f"output would be {grid.width_px}x{grid.height_px} ({mp:.0f} MP), over the "
            f"{opts.max_output_megapixels:.0f} MP limit; use gsd_mm >= {math.ceil(need * 10) / 10} "
            "or split the wall"
        )
    _log(progress, f"Output {grid.width_px}x{grid.height_px} px at {gsd_m * 1000:.2f} mm/px ({gsd_source})")

    # --- stage 5a: exposure gains ---------------------------------------------
    flat_valid = valid.reshape(len(kept), -1)
    gains = solve_gains(samples.reshape(len(kept), -1, 3), flat_valid, sel)

    # --- stage 5b: fine pass, tile by tile ------------------------------------
    phase = time.time()
    out_path = Path(workdir) / "facade_rgba.u8"
    out = np.memmap(out_path, dtype=np.uint8, mode="w+", shape=(grid.height_px, grid.width_px, 4))
    cache = ImageCache(int(opts.image_cache_gb * 1024 ** 3))
    margin = 2 ** (opts.blend_levels + 1)
    tile = opts.tile_px
    tiles = [(r, c) for r in range(0, grid.height_px, tile) for c in range(0, grid.width_px, tile)]
    used_pixels = np.zeros(len(kept), dtype=np.int64)
    used_gsd = [[] for _ in kept]
    for ti, (r0, c0) in enumerate(tiles):
        r1, c1 = min(r0 + tile, grid.height_px), min(c0 + tile, grid.width_px)
        R0, C0, R1, C1 = r0 - margin, c0 - margin, r1 + margin, c1 + margin
        u, v = grid.uv(R0, R1, C0, C1)
        pts = plane.to_world(u, v, depth.sample(u, v))
        ci = np.clip((u / coarse_cell).astype(np.int64), 0, cgrid.width_px - 1)
        ri = np.clip(((plane.height_m - v) / coarse_cell).astype(np.int64), 0, cgrid.height_px - 1)
        coarse = coarse_labels[ri, ci]
        cwin = coarse_labels[max(ri.min() - 1, 0): ri.max() + 2, max(ci.min() - 1, 0): ci.max() + 2]
        tile_cams = [int(k) for k in np.unique(cwin) if k >= 0]
        th, tw = u.shape
        if not tile_cams:
            continue
        imgs, fscores, fvalid = [], [], []
        for k in tile_cams:
            shot = kept[k]
            px, py, d = shot.project(pts)
            sc, g = score_views(shot, pts, plane.w, px, py, d, sel)
            ok = np.isfinite(sc) & zbuffers[shot.name].visible(px, py, d)
            gm = float(np.nanmedian(np.where(ok, g, np.nan))) if ok.any() else gsd_m
            level = int(np.clip(math.floor(math.log2(max(gsd_m / max(gm, 1e-9), 1.0))), 0, 4))
            src = cache.get(shot, level)
            f = 2.0 ** level
            mx = np.where(ok, (px + 0.5) / f - 0.5, -1.0).astype(np.float32)
            my = np.where(ok, (py + 0.5) / f - 0.5, -1.0).astype(np.float32)
            sampled = cv2.remap(src, mx, my, cv2.INTER_CUBIC,
                                borderMode=cv2.BORDER_REPLICATE).astype(np.float32)
            imgs.append(sampled * gains[k][None, None, :].astype(np.float32))
            fscores.append(np.where(ok, sc, -np.inf))
            fvalid.append(ok)
            used_gsd[k].append(gm)
        fscores, fvalid = np.stack(fscores), np.stack(fvalid)
        # coarse choice where that photo really sees the pixel, best fine score otherwise
        idx = np.full(coarse.shape, -1, dtype=np.int64)
        for j, k in enumerate(tile_cams):
            idx[(coarse == k) & fvalid[j]] = j
        need = (idx < 0) & fvalid.any(axis=0)
        if need.any():
            idx[need] = fscores.argmax(axis=0)[need]
        covered_t = idx >= 0
        composite = np.zeros((th, tw, 3), np.float32)
        for j in range(len(tile_cams)):
            composite[idx == j] = imgs[j][idx == j]
        masks = []
        for j in range(len(tile_cams)):
            imgs[j][~fvalid[j]] = composite[~fvalid[j]]
            masks.append((idx == j).astype(np.float32))
        blended = multiband_blend(imgs, masks, opts.blend_levels) if len(tile_cams) > 1 else composite
        core = (slice(margin, margin + (r1 - r0)), slice(margin, margin + (c1 - c0)))
        rgb = np.clip(blended[core], 0, 255).astype(np.uint8)[..., ::-1]  # BGR -> RGB
        alpha = np.where(covered_t[core], 255, 0).astype(np.uint8)
        rgb[alpha == 0] = 0
        out[r0:r1, c0:c1, :3] = rgb
        out[r0:r1, c0:c1, 3] = alpha
        core_idx = idx[core]
        for j, k in enumerate(tile_cams):
            used_pixels[k] += int((core_idx == j).sum())
        if progress and (ti % max(1, len(tiles) // 10) == 0 or ti == len(tiles) - 1):
            _log(progress, f"Blending tiles {ti + 1}/{len(tiles)}")
    out.flush()
    timings["render_seconds"] = round(time.time() - phase, 1)

    coverage = float((out[..., 3] > 0).mean())
    if coverage < 0.98:
        warnings.append(f"{(1 - coverage) * 100:.1f}% of the wall had no unobstructed photo and is transparent")

    cams_used = []
    total_px = max(int(used_pixels.sum()), 1)
    for k, shot in enumerate(kept):
        if used_pixels[k] == 0:
            continue
        cams_used.append({
            "name": shot.name,
            "share": round(used_pixels[k] / total_px, 4),
            "native_gsd_mm": round(float(np.median(used_gsd[k])) * 1000, 3) if used_gsd[k] else None,
            "standoff_m": round(float((shot.center - plane.origin) @ plane.w), 2),
            "gain_bgr": [round(float(g), 4) for g in gains[k]],
        })
    cams_used.sort(key=lambda c: -c["share"])
    timings["total_seconds"] = round(time.time() - t0, 1)

    return {
        "raster_path": str(out_path),
        "grid": grid,
        "gsd_source": gsd_source,
        "native_gsd_mm_median": round(native_m * 1000, 3),
        "coverage": round(coverage, 4),
        "coverage_coarse": round(coverage_coarse, 4),
        "depth": depth.stats(),
        "occluder_points": int(len(occluders)),
        "candidates_considered": n_all,
        "candidates_occlusion_checked": candidates_checked,
        "cameras_used": cams_used,
        "image_decodes": cache.decodes,
        "warnings": warnings,
        "timings": timings,
        "options": _jsonable(asdict(opts)),
    }


def _jsonable(x):
    if isinstance(x, dict):
        return {k: _jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_jsonable(v) for v in x]
    return x
