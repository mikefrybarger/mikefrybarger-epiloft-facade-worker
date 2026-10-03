"""RunPod handler end to end: signed-URL download, render, signed-URL uploads."""
import http.server
import io
import json
import sys
import threading
import types
import zipfile
from pathlib import Path

import pytest

requests = pytest.importorskip("requests")
tifffile = pytest.importorskip("tifffile")

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

progress_log = []
fake_runpod = types.ModuleType("runpod")
fake_runpod.serverless = types.SimpleNamespace(progress_update=lambda job, msg: progress_log.append(msg))
sys.modules["runpod"] = fake_runpod

import synthetic as syn  # noqa: E402
import handler  # noqa: E402


def _zip_project(root: Path) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        for p in root.rglob("*"):
            if p.is_file():
                zf.write(p, Path("task") / p.relative_to(root))
    return buf.getvalue()


class _Server:
    def __init__(self, archive):
        self.archive, self.uploads, self.headers = archive, {}, {}
        srv = self

        class H(http.server.BaseHTTPRequestHandler):
            def log_message(self, *a):
                pass

            def do_GET(self):
                self.send_response(200)
                self.send_header("Content-Length", str(len(srv.archive)))
                self.end_headers()
                self.wfile.write(srv.archive)

            def do_PUT(self):
                n = int(self.headers["Content-Length"])
                srv.uploads[self.path] = self.rfile.read(n)
                srv.headers[self.path] = self.headers.get("Content-Type")
                self.send_response(200)
                self.end_headers()

            def do_POST(self):  # upload link refresh
                body = json.dumps({"ortho_upload_url": srv.url("/up/refreshed.tif")}).encode()
                self.send_response(200)
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

        self.httpd = http.server.ThreadingHTTPServer(("127.0.0.1", 0), H)
        threading.Thread(target=self.httpd.serve_forever, daemon=True).start()

    def url(self, path):
        return f"http://127.0.0.1:{self.httpd.server_address[1]}{path}"


@pytest.fixture(scope="module")
def server(tmp_path_factory):
    root = tmp_path_factory.mktemp("odm")
    syn.build_project(root)
    s = _Server(_zip_project(root))
    yield s
    s.httpd.shutdown()


def test_handler_round_trip(server):
    job = {"id": "job-1", "input": {
        "project_id": "synthetic",
        "wall_name": "south",
        "source_url": server.url("/src.zip"),
        "wall": syn.wall_payload(),
        "gsd_mm": 8,
        "ortho_upload_url": server.url("/up/facade.tif"),
        "sidecar_upload_url": server.url("/up/facade.json"),
        "preview_upload_url": server.url("/up/preview.jpg"),
        "upload_url_refresh_url": server.url("/refresh"),
    }}
    result = handler.handler(job)
    assert result["status"] == "completed"
    assert result["upload_links_refreshed"] is True
    assert "/up/refreshed.tif" in server.uploads           # refreshed link was used
    assert "/up/facade.tif" not in server.uploads
    assert server.headers["/up/refreshed.tif"] == "image/tiff"
    rgba = tifffile.imread(io.BytesIO(server.uploads["/up/refreshed.tif"]))
    meta = json.loads(server.uploads["/up/facade.json"])
    assert rgba.shape[:2] == (meta["image"]["height_px"], meta["image"]["width_px"])
    assert meta["project_id"] == "synthetic" and meta["wall_name"] == "south"
    assert meta["plane"]["mesh"]["bottom_left"]  # mesh-frame corners echoed back for Studio
    assert server.uploads["/up/preview.jpg"][:2] == b"\xff\xd8"
    assert any("Blending" in m for m in progress_log)


def test_handler_validates_input(server):
    with pytest.raises(ValueError, match="ortho_upload_url"):
        handler.handler({"input": {"source_url": server.url("/src.zip")}})
    with pytest.raises(ValueError, match="mesh_origin"):
        wall = syn.wall_payload()
        del wall["mesh_origin"]
        handler.handler({"input": {"source_url": server.url("/src.zip"), "wall": wall,
                                   "ortho_upload_url": server.url("/up/x.tif")}})
    with pytest.raises(ValueError, match="unknown option"):
        handler.handler({"input": {"source_url": server.url("/src.zip"), "wall": syn.wall_payload(),
                                   "ortho_upload_url": server.url("/up/x.tif"),
                                   "options": {"blend_levle": 3}}})
