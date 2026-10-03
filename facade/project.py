"""Locate the pieces of an extracted ODM project (same ZIP the splat worker takes)."""
from __future__ import annotations

import json
import re
import zipfile
from dataclasses import dataclass
from pathlib import Path

from .cameras import load_reconstruction
from .georef import resolve_frame
from .images import image_size
from .pointcloud import load_geo_mesh_vertices, load_point_cloud


def safe_extract(zip_path: Path, dest: Path):
    """Extract a ZIP, refusing entries that escape dest (zip-slip)."""
    dest = dest.resolve()
    with zipfile.ZipFile(zip_path) as zf:
        for member in zf.infolist():
            target = (dest / member.filename).resolve()
            if target != dest and dest not in target.parents:
                raise ValueError(f"Unsafe path in ZIP: {member.filename}")
        zf.extractall(dest)


def _basename(path_str: str) -> str:
    return re.split(r"[\\/]", path_str)[-1]


def find_project_root(extract_dir: Path) -> Path:
    candidates = [p.parent for p in extract_dir.rglob("opensfm")
                  if p.is_dir() and "__MACOSX" not in p.parts]
    if not candidates:
        raise RuntimeError("No opensfm/ folder found in the source dataset")
    candidates.sort(key=lambda p: (len(p.relative_to(extract_dir).parts), str(p)))
    root = candidates[0]
    if not (root / "opensfm" / "reconstruction.json").is_file():
        raise RuntimeError("Missing opensfm/reconstruction.json")
    return root


@dataclass
class Project:
    root: Path
    shots: dict
    offset_e: float
    offset_n: float
    epsg: int | None
    source_frame: str
    missing_images: list
    points: object = None          # dense cloud (N,3) in the same frame as the shots, or None
    points_source: str | None = None
    frame_report: dict | None = None


def parse_coords(path: Path):
    """(epsg, offset_e, offset_n) from ODM coords.txt."""
    lines = path.read_text().splitlines()
    epsg = None
    header = lines[0].strip() if lines else ""
    m = re.match(r"WGS84\s+UTM\s+(\d{1,2})\s*([NS])", header, re.IGNORECASE)
    if m:
        epsg = (32600 if m.group(2).upper() == "N" else 32700) + int(m.group(1))
    else:
        m = re.match(r"EPSG:(\d+)", header, re.IGNORECASE)
        if m:
            epsg = int(m.group(1))
    e, n = (float(x) for x in lines[1].split()[:2])
    return epsg, e, n


def load_project(search_dir: Path) -> Project:
    root = find_project_root(search_dir)
    shots, sparse = load_reconstruction(root / "opensfm" / "reconstruction.json")

    index = {}
    for p in search_dir.rglob("*"):
        if p.is_file() and "__MACOSX" not in p.parts:
            index.setdefault(p.name, []).append(p)
    images_dir = root / "images"
    missing = []
    for name, shot in shots.items():
        base = _basename(name)
        preferred = images_dir / base
        if preferred.is_file():
            shot.image_path = preferred
        elif len(index.get(base, [])) == 1:
            shot.image_path = index[base][0]
        else:
            missing.append(base)
            continue
        w, h = image_size(shot.image_path)
        cw, ch = shot.camera.width, shot.camera.height
        if (w >= h) != (cw >= ch) and w != h:
            raise RuntimeError(
                f"{base} is {w}x{h} but its camera model is {cw}x{ch}: the image "
                "orientation differs from what OpenSfM solved. Re-export the photos "
                "without EXIF rotation applied."
            )
        if abs(w / h - cw / ch) > 0.01:
            raise RuntimeError(f"{base} aspect {w}x{h} does not match camera {cw}x{ch}")
        shot.image_w, shot.image_h = w, h

    usable = {k: s for k, s in shots.items() if s.image_path is not None}

    coords = root / "odm_georeferencing" / "coords.txt"
    marker = (root / "opensfm" / "reconstruction.topocentric.json").is_file()
    ref_lla = None
    ref_path = root / "opensfm" / "reference_lla.json"
    if ref_path.is_file():
        try:
            ref = json.loads(ref_path.read_text())
            ref_lla = {"lat": float(ref["latitude"]), "lon": float(ref["longitude"]),
                       "alt": float(ref.get("altitude", 0.0))}
        except (ValueError, KeyError, TypeError) as exc:
            print(f"could not read reference_lla.json: {exc}", flush=True)

    if coords.is_file():
        epsg, oe, on = parse_coords(coords)
        frame = "odm_utm_offset"
    else:
        epsg, oe, on = None, 0.0, 0.0
        frame = "opensfm_topocentric"

    points, points_src, kind = load_point_cloud(root, oe, on)
    report = None
    if frame == "odm_utm_offset":
        reference, ref_src = (points, points_src) if kind == "laz" else load_geo_mesh_vertices(root)
        report, transform = resolve_frame(usable, sparse, reference, ref_lla, epsg, oe, on, marker)
        report["reference"] = ref_src
        if kind == "ply":
            points = transform(points)  # the PLY was in the poses' original frame
        elif points is None and reference is not None:
            # no dense cloud at all: the textured mesh still gives depth and occlusion
            points, points_src = reference, ref_src
        print("pose frame: " + json.dumps({k: v for k, v in report.items() if k != "warnings"}), flush=True)
    return Project(root=root, shots=usable, offset_e=oe, offset_n=on, epsg=epsg,
                   source_frame=frame, missing_images=missing, points=points,
                   points_source=points_src, frame_report=report)
