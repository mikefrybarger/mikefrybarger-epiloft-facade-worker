"""Dense point cloud loading (XYZ only), in the OpenSfM frame.

Preferred source is ODM's ``odm_filterpoints/point_cloud.ply``, which is in the
same offset frame as ``reconstruction.json``. The georeferenced LAZ is in full
UTM and gets the coords.txt offset subtracted.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

_PLY_TYPES = {
    "char": "i1", "int8": "i1", "uchar": "u1", "uint8": "u1",
    "short": "i2", "int16": "i2", "ushort": "u2", "uint16": "u2",
    "int": "i4", "int32": "i4", "uint": "u4", "uint32": "u4",
    "float": "f4", "float32": "f4", "double": "f8", "float64": "f8",
}

PLY_CANDIDATES = (
    "odm_filterpoints/point_cloud.ply",
    "odm_meshing/odm_mesh.ply",  # vertices only; coarser, last resort
)
LAZ_CANDIDATES = (
    "odm_georeferencing/odm_georeferenced_model.laz",
    "odm_georeferencing/odm_georeferenced_model.las",
)


def read_ply_xyz(path: Path) -> np.ndarray:
    """Vertex x, y, z from a PLY file (ascii or binary little/big endian)."""
    with open(path, "rb") as f:
        if f.readline().strip() != b"ply":
            raise ValueError(f"{path} is not a PLY file")
        fmt = None
        elements = []  # (name, count, [(prop, dtype) or ('list', ...)])
        while True:
            line = f.readline()
            if not line:
                raise ValueError(f"{path}: truncated PLY header")
            parts = line.decode("ascii", "replace").split()
            if not parts:
                continue
            if parts[0] == "format":
                fmt = parts[1]
            elif parts[0] == "element":
                elements.append([parts[1], int(parts[2]), []])
            elif parts[0] == "property":
                if parts[1] == "list":
                    elements[-1][2].append(("__list__", parts[2], parts[3], parts[4]))
                else:
                    elements[-1][2].append((parts[2], _PLY_TYPES[parts[1]]))
            elif parts[0] == "end_header":
                break

        if not elements or elements[0][0] != "vertex":
            raise ValueError(f"{path}: first PLY element is not 'vertex'")
        name, count, props = elements[0]
        if any(p[0] == "__list__" for p in props):
            raise ValueError(f"{path}: list property in vertex element is not supported")
        names = [p[0] for p in props]
        for axis in ("x", "y", "z"):
            if axis not in names:
                raise ValueError(f"{path}: vertex element has no '{axis}'")

        if fmt == "ascii":
            rows = [f.readline().split() for _ in range(count)]
            arr = np.asarray(rows, dtype=np.float64)
            idx = [names.index(a) for a in ("x", "y", "z")]
            return arr[:, idx]
        endian = "<" if fmt == "binary_little_endian" else ">"
        dtype = np.dtype([(p[0], endian + p[1]) for p in props])
        data = np.fromfile(f, dtype=dtype, count=count)
        return np.stack([data["x"], data["y"], data["z"]], axis=-1).astype(np.float64)


def read_laz_xyz(path: Path, offset_e: float, offset_n: float) -> np.ndarray:
    try:
        import laspy  # noqa: PLC0415
    except ImportError as exc:  # pragma: no cover - depends on image
        raise RuntimeError("laspy is not installed; cannot read LAZ") from exc
    las = laspy.read(str(path))
    return np.stack([np.asarray(las.x) - offset_e, np.asarray(las.y) - offset_n, np.asarray(las.z)], axis=-1)


def find_point_cloud(project_root: Path, prefer_laz: bool = True):
    """The georeferenced LAZ comes first: its frame (absolute UTM) is never in
    doubt, while the PLY is in whatever frame the reconstruction was in."""
    ply = [(project_root / rel, "ply") for rel in PLY_CANDIDATES]
    laz = [(project_root / rel, "laz") for rel in LAZ_CANDIDATES]
    for p, kind in (laz + ply if prefer_laz else ply + laz):
        if p.is_file():
            return p, kind
    return None, None


def load_point_cloud(project_root: Path, offset_e=0.0, offset_n=0.0, prefer_laz: bool = True):
    """Returns (points (N,3), source path, kind) or (None, None, None).

    LAZ comes back in the offset frame; PLY comes back as stored.
    """
    path, kind = find_point_cloud(project_root, prefer_laz)
    if path is None:
        return None, None, None
    if kind == "ply":
        pts = read_ply_xyz(path)
    else:
        pts = read_laz_xyz(path, offset_e, offset_n)
    pts = pts[np.all(np.isfinite(pts), axis=1)]
    return pts, str(path.relative_to(project_root)), kind


GEO_MESH_CANDIDATES = (
    "odm_texturing/odm_textured_model_geo.obj",
    "odm_texturing_25d/odm_textured_model_geo.obj",
)


def read_obj_vertices(path: Path, max_vertices: int = 5_000_000) -> np.ndarray:
    """Vertex positions from a Wavefront OBJ (``v x y z`` lines only)."""
    out = []
    with open(path, "rb") as f:
        for line in f:
            if line.startswith(b"v "):
                parts = line.split()
                out.append((float(parts[1]), float(parts[2]), float(parts[3])))
                if len(out) >= max_vertices:
                    break
    return np.asarray(out, dtype=np.float64).reshape(-1, 3)


def load_geo_mesh_vertices(project_root: Path):
    """ODM's georeferenced textured mesh, already in the offset frame.

    This is the geometry Studio displays, so it is the best thing to check the
    pose frame against when the archive has no georeferenced LAZ.
    """
    for rel in GEO_MESH_CANDIDATES:
        p = project_root / rel
        if p.is_file():
            v = read_obj_vertices(p)
            v = v[np.all(np.isfinite(v), axis=1)]
            if len(v):
                return v, rel
    return None, None
