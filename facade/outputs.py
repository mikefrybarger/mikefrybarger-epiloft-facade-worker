"""Stage 6: deliverables.

* ``facade.tif``  tiled BigTIFF, RGBA, lossless. Transparent where no photo saw
  the wall. Resolution tags carry the true scale, so CAD and image tools that
  read DPI get real-world size.
* ``facade.json`` sidecar: the plane, the pixel -> wall -> world maths, scale,
  photos used, warnings. This (not GeoTIFF tags) is the georeferencing: a
  vertical wall has no meaningful map projection.
* ``facade.jpg``  full-resolution JPEG, alpha flattened onto white. The file
  people actually open: every viewer, browser and phone reads it.
* ``facade_preview.jpg``  long edge <= 4096, alpha flattened onto white.
* ``facade_tiles.zip``  Deep Zoom (DZI) pyramid, only when pyvips is present.
"""
from __future__ import annotations

import json
import zipfile
from pathlib import Path

import cv2
import numpy as np

SIDECAR_VERSION = 1
PREVIEW_LONG_EDGE = 4096
JPEG_MAX_SIDE = 65_500          # JPEG's hard limit is 65,535 px per side
BIGTIFF_THRESHOLD = 3_500_000_000


def write_tiff(raster: np.ndarray, path: Path, gsd_m: float, description: str = ""):
    """Tiled RGBA TIFF. Classic TIFF (not BigTIFF) whenever it fits, and LZW,
    the compression the widest range of viewers and CAD tools read."""
    import tifffile  # noqa: PLC0415

    px_per_cm = 1.0 / (gsd_m * 100.0)
    tifffile.imwrite(
        path, raster, bigtiff=raster.nbytes > BIGTIFF_THRESHOLD, tile=(512, 512), photometric="rgb",
        extrasamples=["unassalpha"], compression="lzw",
        resolution=(px_per_cm, px_per_cm), resolutionunit="CENTIMETER",
        description=description, software="epiloft-facade-worker", metadata=None,
    )


def write_full_jpeg(raster: np.ndarray, path: Path, quality: int = 92):
    """Full-resolution JPEG (downscaled only past JPEG's 65,535 px limit).

    Returns (width, height, scale) where scale < 1 means it was reduced."""
    h, w = raster.shape[:2]
    if max(h, w) > JPEG_MAX_SIDE:
        tw, th = write_preview(raster, path, long_edge=JPEG_MAX_SIDE, quality=quality)
        return tw, th, tw / w
    img = np.empty((h, w, 3), dtype=np.uint8)
    for y0 in range(0, h, 2048):
        band = np.asarray(raster[y0:y0 + 2048]).astype(np.float32)
        a = band[..., 3:4] / 255.0
        img[y0:y0 + 2048] = np.clip(band[..., :3] * a + 255.0 * (1.0 - a), 0, 255).astype(np.uint8)[..., ::-1]
    if not cv2.imwrite(str(path), img, [cv2.IMWRITE_JPEG_QUALITY, quality,
                                        cv2.IMWRITE_JPEG_OPTIMIZE, 1]):
        raise RuntimeError("could not write the full-resolution JPEG")
    return w, h, 1.0


def write_preview(raster: np.ndarray, path: Path, long_edge: int = PREVIEW_LONG_EDGE, quality: int = 88):
    h, w = raster.shape[:2]
    scale = min(1.0, long_edge / max(h, w))
    tw, th = max(1, round(w * scale)), max(1, round(h * scale))
    # Resize in horizontal bands so a huge raster is never fully in memory twice.
    band_out = max(1, 1024)
    rows = []
    for y0 in range(0, th, band_out):
        y1 = min(th, y0 + band_out)
        sy0, sy1 = int(y0 / scale), min(h, int(np.ceil(y1 / scale)))
        band = np.asarray(raster[sy0:sy1]).astype(np.float32)
        a = band[..., 3:4] / 255.0
        flat = band[..., :3] * a + 255.0 * (1.0 - a)
        rows.append(cv2.resize(flat, (tw, y1 - y0), interpolation=cv2.INTER_AREA))
    img = np.clip(np.concatenate(rows, axis=0), 0, 255).astype(np.uint8)
    cv2.imwrite(str(path), img[..., ::-1], [cv2.IMWRITE_JPEG_QUALITY, quality])
    return tw, th


def write_tiles(tiff_path: Path, zip_path: Path) -> bool:
    """Deep Zoom pyramid as a ZIP. Returns False when pyvips is unavailable."""
    try:
        import pyvips  # noqa: PLC0415
    except (ImportError, OSError):
        return False
    image = pyvips.Image.new_from_file(str(tiff_path), access="sequential")
    base = zip_path.with_suffix("")
    image.dzsave(str(base), suffix=".png", tile_size=512, overlap=1, container="zip")
    # libvips versions differ on whether ".zip" is appended to the name.
    for produced in (base.with_suffix(".zip"), base):
        if produced.is_file():
            if produced != zip_path:
                produced.rename(zip_path)
            break
    else:
        raise RuntimeError("libvips did not write the tile pyramid ZIP")
    with zipfile.ZipFile(zip_path) as zf:
        if not any(n.endswith(".dzi") for n in zf.namelist()):
            raise RuntimeError("tile pyramid was written without a .dzi descriptor")
    return True


def build_sidecar(result: dict, plane, frame_ctx, project, extra: dict) -> dict:
    grid = result["grid"]
    return {
        "version": SIDECAR_VERSION,
        "kind": "epiloft.facade_ortho",
        "image": {
            "width_px": grid.width_px,
            "height_px": grid.height_px,
            "gsd_m": grid.gsd_m,
            "gsd_mm": round(grid.gsd_m * 1000, 4),
            "gsd_source": result["gsd_source"],
            "native_gsd_mm_median": result["native_gsd_mm_median"],
            "coverage": result["coverage"],
            "alpha": "0 = no unobstructed photo of that spot",
        },
        "pixel_to_wall": {
            "convention": "x right, y down, (0,0) = top-left corner of the image",
            "u_m": "x * gsd_m",
            "v_m": "height_m - y * gsd_m",
            "note": "distances measured on the image are true on the wall plane; "
                    "surface relief (trim, recesses) is in depth.offset_*",
        },
        "wall_to_world": {
            "formula": "P = origin + u*U + v*V + w*W (opensfm frame, metres)",
            "source_frame": project.source_frame,
            "utm": ({"epsg": project.epsg, "offset_e": project.offset_e, "offset_n": project.offset_n}
                    if project.source_frame == "odm_utm_offset" else None),
            "pose_frame": project.frame_report,
        },
        "plane": plane.describe(frame_ctx),
        "depth": result["depth"],
        "cameras_used": result["cameras_used"],
        "candidates_considered": result["candidates_considered"],
        "warnings": result["warnings"],
        "timings": result["timings"],
        "options": result["options"],
        **extra,
    }


def write_json(obj: dict, path: Path):
    path.write_text(json.dumps(obj, indent=2))
