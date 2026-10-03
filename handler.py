"""RunPod Serverless entry point and local CLI for the Epiloft facade worker.

RunPod:   python3 -u handler.py
Local:    python3 handler.py --local path/to/odm_project_or_zip --wall wall.json --out ./out
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

from facade.geometry import FrameContext, WallPlane
from facade.outputs import (build_sidecar, write_full_jpeg, write_json, write_preview, write_tiff,
                            write_tiles)
from facade.pipeline import FacadeOptions, run_facade
from facade.project import load_project, safe_extract
from facade.transfer import download, refresh_upload_urls, require_http_url, upload

WORKER_VERSION = "2026-10-03.6"
GB = 1024 ** 3
DISK_HEADROOM_FACTOR = 2.5

# payload key -> (output file, content type, required)
UPLOADS = {
    "ortho_upload_url": ("facade.tif", "image/tiff", True),
    "jpeg_upload_url": ("facade.jpg", "image/jpeg", False),
    "sidecar_upload_url": ("facade.json", "application/json", False),
    "preview_upload_url": ("facade_preview.jpg", "image/jpeg", False),
    "tiles_upload_url": ("facade_tiles.zip", "application/zip", False),
}


def _progress_for(job):
    try:
        import runpod  # noqa: PLC0415
    except ImportError:
        return None
    if job is None:
        return None
    return lambda msg: runpod.serverless.progress_update(job, msg)


def parse_wall(spec: dict, project) -> tuple[WallPlane, FrameContext | None]:
    """Turn the payload's wall block into a plane in the OpenSfM frame.

    spec = {
      "frame": "mesh" | "opensfm",
      "corners": {"bottom_left": [x,y,z], "bottom_right": [...], "top_left": [...]},
      "mesh_origin": {"e": .., "n": .., "z": ..},   # required for frame "mesh"
      "view_from": [x, y, z]     # optional but recommended: where the viewer stood when
                                 # picking the wall (Studio camera position), same frame
                                 # as the corners. Decides which face is the outside.
    }
    """
    if not isinstance(spec, dict):
        raise ValueError("wall must be an object")
    corners = spec.get("corners") or {}
    for key in ("bottom_left", "bottom_right", "top_left"):
        if key not in corners or len(corners[key]) != 3:
            raise ValueError(f"wall.corners.{key} must be [x, y, z]")
    frame = str(spec.get("frame", "mesh")).lower()
    ctx = None
    if frame == "mesh":
        if project.source_frame != "odm_utm_offset":
            raise ValueError("wall.frame 'mesh' needs a georeferenced ODM run "
                             "(reconstruction.topocentric.json + coords.txt); send frame 'opensfm'")
        origin = spec.get("mesh_origin")
        if not isinstance(origin, dict) or "e" not in origin or "n" not in origin or "z" not in origin:
            raise ValueError("wall.mesh_origin {e, n, z} is required for frame 'mesh' "
                             "(the same mesh_origin Studio uses to place the model)")
        ctx = FrameContext.from_payload((project.offset_e, project.offset_n), origin)
        pts = [ctx.mesh_to_opensfm(corners[k]) for k in ("bottom_left", "bottom_right", "top_left")]
    elif frame == "opensfm":
        pts = [corners[k] for k in ("bottom_left", "bottom_right", "top_left")]
        if project.source_frame == "odm_utm_offset" and isinstance(spec.get("mesh_origin"), dict):
            ctx = FrameContext.from_payload((project.offset_e, project.offset_n), spec["mesh_origin"])
    else:
        raise ValueError("wall.frame must be 'mesh' or 'opensfm'")
    plane = WallPlane.from_corners(*pts)
    view = spec.get("view_from")
    if view is not None:
        if not isinstance(view, (list, tuple)) or len(view) != 3:
            raise ValueError("wall.view_from must be [x, y, z] in the same frame as the corners")
        view = ctx.mesh_to_opensfm(view) if frame == "mesh" else np.asarray(view, dtype=np.float64)
        if abs(float((view - plane.origin) @ plane.w)) < 0.25:
            raise ValueError("wall.view_from lies in the wall plane; send the Studio camera position")
        plane.view_from = view
    return plane, ctx


def produce(project_dir: Path, job_input: dict, out_dir: Path, progress=None) -> dict:
    """Run the pipeline and write every deliverable into out_dir."""
    started = time.time()
    project = load_project(project_dir)
    if project.missing_images:
        print(f"note: {len(project.missing_images)} solved shots have no image in the dataset", flush=True)
    plane, ctx = parse_wall(job_input.get("wall"), project)

    opts_in = dict(job_input.get("options") or {})
    gsd = job_input.get("gsd_mm", "native")
    if gsd not in (None, "native"):
        gsd = float(gsd)
        if not 0.2 <= gsd <= 100:
            raise ValueError("gsd_mm must be between 0.2 and 100, or 'native'")
        opts_in["gsd_mm"] = gsd
    opts = FacadeOptions.from_payload(opts_in)

    out_dir.mkdir(parents=True, exist_ok=True)
    result = run_facade(project, plane, opts, out_dir, progress)
    grid = result["grid"]

    outputs_started = time.time()
    raster = np.memmap(result["raster_path"], dtype=np.uint8, mode="r",
                       shape=(grid.height_px, grid.width_px, 4))
    try:
        sidecar = build_sidecar(result, plane, ctx, project, {
            "project_id": job_input.get("project_id"),
            "wall_name": job_input.get("wall_name"),
            "worker_version": WORKER_VERSION,
            "files": {"ortho": "facade.tif", "jpeg": "facade.jpg", "preview": "facade_preview.jpg"},
        })
        if progress:
            progress("Writing TIFF")
        write_tiff(raster, out_dir / "facade.tif", grid.gsd_m,
                   description=json.dumps({"kind": sidecar["kind"], "gsd_m": grid.gsd_m,
                                           "plane": sidecar["plane"]["opensfm"]}))
        jw, jh, jscale = write_full_jpeg(raster, out_dir / "facade.jpg")
        sidecar["jpeg"] = {"width_px": jw, "height_px": jh, "scale": round(jscale, 6),
                           "gsd_mm": round(grid.gsd_m * 1000 / jscale, 4)}
        if jscale < 1:
            sidecar["warnings"].append(
                f"facade.jpg was reduced to {jw}x{jh} (JPEG's size limit); facade.tif is full resolution")
        pw, ph = write_preview(raster, out_dir / "facade_preview.jpg")
        sidecar["preview"] = {"width_px": pw, "height_px": ph}
        want_tiles = bool(job_input.get("tiles_upload_url")) or bool(job_input.get("make_tiles"))
        if want_tiles:
            if progress:
                progress("Building deep-zoom tiles")
            if write_tiles(out_dir / "facade.tif", out_dir / "facade_tiles.zip"):
                sidecar["files"]["tiles"] = "facade_tiles.zip"
            else:
                sidecar["warnings"].append("deep-zoom tiles skipped: pyvips is not installed")
    finally:
        del raster
        Path(result["raster_path"]).unlink(missing_ok=True)
    sidecar["timings"]["outputs_seconds"] = round(time.time() - outputs_started, 1)
    sidecar["timings"]["job_seconds"] = round(time.time() - started, 1)
    write_json(sidecar, out_dir / "facade.json")
    return sidecar


def handler(job):
    started = time.time()
    job_input = job.get("input") or {}
    source_url = require_http_url(job_input.get("source_url"), "source_url")
    require_http_url(job_input.get("ortho_upload_url"), "ortho_upload_url")
    for key in UPLOADS:
        if job_input.get(key):
            require_http_url(job_input[key], key)
    method = str(job_input.get("output_method", "PUT")).upper()
    headers = job_input.get("output_headers") or {}
    progress = _progress_for(job)

    with tempfile.TemporaryDirectory(prefix="epiloft-facade-") as tmp:
        workdir = Path(tmp)
        src = workdir / "source.zip"
        if progress:
            progress("Downloading source dataset")
        size = download(source_url, src, job_input.get("source_headers") or {})
        extract = workdir / "project"
        extract.mkdir()
        free = shutil.disk_usage(extract).free
        if free < size * DISK_HEADROOM_FACTOR:
            raise RuntimeError(f"Not enough disk: need {size * DISK_HEADROOM_FACTOR / GB:.1f} GB, "
                               f"have {free / GB:.1f} GB. Increase container disk.")
        if progress:
            progress("Extracting dataset")
        safe_extract(src, extract)
        src.unlink()
        out_dir = workdir / "out"
        sidecar = produce(extract, job_input, out_dir, progress)
        shutil.rmtree(extract, ignore_errors=True)

        urls = {k: job_input.get(k) for k in UPLOADS}
        refreshed = refresh_upload_urls(job_input.get("upload_url_refresh_url"))
        if refreshed:
            for k in UPLOADS:
                if urls.get(k) and refreshed.get(k):
                    urls[k] = refreshed[k]
        uploaded = {}
        for key, (name, ctype, _required) in UPLOADS.items():
            path = out_dir / name
            if urls.get(key) and path.is_file():
                if progress:
                    progress(f"Uploading {name}")
                upload(urls[key], path, method=method, headers=headers, content_type=ctype)
                uploaded[name] = path.stat().st_size

    result = {
        "project_id": job_input.get("project_id"),
        "status": "completed",
        "worker_version": WORKER_VERSION,
        "uploaded": uploaded,
        "upload_links_refreshed": bool(refreshed),
        "total_seconds": round(time.time() - started, 1),
        "facade": sidecar,
    }
    print(json.dumps({k: v for k, v in result.items() if k != "facade"}), flush=True)
    return result


def _cli(argv):
    ap = argparse.ArgumentParser(description="Render a facade ortho from a local ODM project.")
    ap.add_argument("--local", required=True, help="ODM project folder or its ZIP")
    ap.add_argument("--wall", required=True, help="JSON file: the payload's wall block (or a full payload)")
    ap.add_argument("--out", required=True, help="output folder")
    ap.add_argument("--gsd-mm", default=None, help="mm per pixel, or 'native' (default)")
    ap.add_argument("--tiles", action="store_true", help="also build the deep-zoom ZIP")
    args = ap.parse_args(argv)

    spec = json.loads(Path(args.wall).read_text())
    job_input = spec if "wall" in spec else {"wall": spec}
    if args.gsd_mm:
        job_input["gsd_mm"] = args.gsd_mm
    if args.tiles:
        job_input["make_tiles"] = True
    src = Path(args.local)
    with tempfile.TemporaryDirectory(prefix="epiloft-facade-") as tmp:
        if src.is_file():
            safe_extract(src, Path(tmp))
            project_dir = Path(tmp)
        else:
            project_dir = src
        sidecar = produce(project_dir, job_input, Path(args.out), progress=lambda m: None)
    img = sidecar["image"]
    print(f"\nwrote {args.out}/facade.tif  {img['width_px']}x{img['height_px']} px "
          f"at {img['gsd_mm']} mm/px, coverage {img['coverage']:.1%}")
    for w in sidecar["warnings"]:
        print(f"warning: {w}")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        _cli(sys.argv[1:])
    else:
        import runpod  # noqa: PLC0415

        runpod.serverless.start({"handler": handler})
