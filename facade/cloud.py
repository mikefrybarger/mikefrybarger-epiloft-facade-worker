"""Streaming access to the dense geometry, so memory scales with one wall, not the site.

A 1,455-photo ODM run produces a georeferenced LAZ with hundreds of millions
of points; reading it whole (laspy's record plus float64 copies) does not fit
in a 32 GB worker. Nothing here needs the whole cloud at once:

* the pose-frame check needs a few million points spread over the site
  -> :meth:`CloudSource.sample`
* the facade needs every point on the wall surface, plus a capped sample of
  anything between the wall and the cameras -> :meth:`CloudSource.wall_region`

Both stream the file in chunks (LAZ via laspy's chunk iterator, binary PLY
via a memory map, OBJ line by line) and keep only what they need.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np

from .pointcloud import _PLY_TYPES, GEO_MESH_CANDIDATES, LAZ_CANDIDATES, PLY_CANDIDATES

CHUNK_POINTS = 4_000_000


def _ply_layout(path: Path):
    """(format, vertex count, numpy dtype, header byte length) of a PLY file."""
    with open(path, "rb") as f:
        if f.readline().strip() != b"ply":
            raise ValueError(f"{path} is not a PLY file")
        fmt, count, props, in_vertex = None, 0, [], False
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
                in_vertex = parts[1] == "vertex" and not props
                if in_vertex:
                    count = int(parts[2])
            elif parts[0] == "property" and in_vertex:
                if parts[1] == "list":
                    raise ValueError(f"{path}: list property in vertex element is not supported")
                props.append((parts[2], _PLY_TYPES[parts[1]]))
            elif parts[0] == "end_header":
                header_len = f.tell()
                break
    endian = "<" if fmt == "binary_little_endian" else ">"
    dtype = np.dtype([(n, endian + t) for n, t in props]) if fmt != "ascii" else None
    return fmt, count, dtype, header_len


def _open_laz(path: Path):
    """laspy reader, with multi-threaded LAZ decompression when available."""
    import laspy  # noqa: PLC0415

    if str(path).lower().endswith(".laz"):
        backend = getattr(laspy.LazBackend, "LazrsParallel", None)
        if backend is not None and backend.is_available():
            return laspy.open(str(path), laz_backend=backend)
    return laspy.open(str(path))


@dataclass
class CloudSource:
    kind: str                     # "laz" | "ply" | "mesh"
    path: Path
    src: str                      # path relative to the project, for reports
    offset_e: float = 0.0
    offset_n: float = 0.0
    transform: Callable | None = None   # applied to every chunk (PLY in a moved frame)
    chunk_points: int = CHUNK_POINTS

    # -- raw chunk stream ---------------------------------------------------
    def total_points(self) -> int | None:
        if self.kind == "laz":
            with _open_laz(self.path) as f:
                return int(f.header.point_count)
        if self.kind == "ply":
            return _ply_layout(self.path)[1]
        return None

    def _raw_chunks(self):
        if self.kind == "laz":
            with _open_laz(self.path) as f:
                for pts in f.chunk_iterator(self.chunk_points):
                    yield np.stack([np.asarray(pts.x, np.float64) - self.offset_e,
                                    np.asarray(pts.y, np.float64) - self.offset_n,
                                    np.asarray(pts.z, np.float64)], axis=-1)
        elif self.kind == "ply":
            fmt, count, dtype, header_len = _ply_layout(self.path)
            if fmt == "ascii":
                from .pointcloud import read_ply_xyz  # noqa: PLC0415

                yield read_ply_xyz(self.path)
                return
            mm = np.memmap(self.path, dtype=dtype, mode="r", offset=header_len, shape=(count,))
            for i in range(0, count, self.chunk_points):
                c = mm[i:i + self.chunk_points]
                yield np.stack([c["x"], c["y"], c["z"]], axis=-1).astype(np.float64)
        else:  # mesh: OBJ vertices, streamed line by line
            buf = []
            with open(self.path, "rb") as f:
                for line in f:
                    if line.startswith(b"v "):
                        p = line.split()
                        buf.append((float(p[1]), float(p[2]), float(p[3])))
                        if len(buf) >= self.chunk_points:
                            yield np.asarray(buf, np.float64)
                            buf = []
            if buf:
                yield np.asarray(buf, np.float64)

    def chunks(self):
        for c in self._raw_chunks():
            c = c[np.all(np.isfinite(c), axis=1)]
            if self.transform is not None:
                c = self.transform(c)
            yield c

    # -- what the pipeline asks for ------------------------------------------
    def sample(self, max_points: int, seed: int = 0) -> np.ndarray:
        """A uniform random sample of at most max_points points."""
        rng = np.random.default_rng(seed)
        total = self.total_points()
        keep_p = 1.0 if not total else min(1.0, max_points / total)
        kept = _Capped(max_points, rng, keep_p)
        for c in self.chunks():
            kept.add(c)
        return kept.result()

    def wall_region(self, plane, *, depth_front_m: float, depth_back_m: float, reach_m: float,
                    max_surface: int = 40_000_000, max_occluders: int = 8_000_000, seed: int = 0):
        """(surface, occluders) points in the shots' frame.

        surface: everything within the wall rectangle and the depth search
        band, at full density (it is what the depth map is built from).
        occluders: anything else between the wall and the farthest camera,
        sampled down to max_occluders.
        """
        rng = np.random.default_rng(seed)
        surface = _Capped(max_surface, rng)
        occ = _Capped(max_occluders, rng)
        wm, hm = plane.width_m, plane.height_m
        for c in self.chunks():
            uvw = plane.to_wall(c)
            u, v, w = uvw[:, 0], uvw[:, 1], uvw[:, 2]
            on_wall = ((u >= -0.5) & (u <= wm + 0.5) & (v >= -0.5) & (v <= hm + 0.5)
                       & (w >= -depth_back_m - 0.1) & (w <= depth_front_m + 0.1))
            between = (~on_wall & (w >= -depth_back_m) & (w <= reach_m)
                       & (u >= -reach_m) & (u <= wm + reach_m) & (v >= -reach_m) & (v <= hm + reach_m))
            surface.add(c[on_wall])
            occ.add(c[between])
        return surface.result(), occ.result(), {"surface_capped": surface.halvings > 0,
                                                "occluders_sampled": occ.keep_p < 1.0}


class _Capped:
    """Accumulate points, keeping at most ~cap by thinning uniformly as needed."""

    def __init__(self, cap: int, rng, keep_p: float = 1.0):
        self.cap, self.rng, self.keep_p = cap, rng, keep_p
        self.parts, self.n, self.halvings = [], 0, 0

    def add(self, pts: np.ndarray):
        if len(pts) == 0:
            return
        if self.keep_p < 1.0:
            pts = pts[self.rng.random(len(pts)) < self.keep_p]
        self.parts.append(pts)
        self.n += len(pts)
        while self.n > 2 * self.cap:
            self.parts = [p[self.rng.random(len(p)) < 0.5] for p in self.parts]
            self.n = sum(len(p) for p in self.parts)
            self.keep_p *= 0.5
            self.halvings += 1

    def result(self) -> np.ndarray:
        if not self.parts:
            return np.zeros((0, 3))
        out = np.concatenate(self.parts)
        if len(out) > self.cap:
            out = out[self.rng.choice(len(out), self.cap, replace=False)]
        return out


def find_cloud(project_root: Path, offset_e: float, offset_n: float) -> CloudSource | None:
    """The best dense geometry in the project: LAZ, then PLY, then textured mesh."""
    for rel in LAZ_CANDIDATES:
        if (project_root / rel).is_file():
            return CloudSource("laz", project_root / rel, rel, offset_e, offset_n)
    for rel in PLY_CANDIDATES:
        if (project_root / rel).is_file():
            return CloudSource("ply", project_root / rel, rel)
    return find_mesh(project_root)


def find_mesh(project_root: Path) -> CloudSource | None:

    for rel in GEO_MESH_CANDIDATES:
        if (project_root / rel).is_file():
            return CloudSource("mesh", project_root / rel, rel)
    return None
