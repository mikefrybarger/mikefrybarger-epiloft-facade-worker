"""Signed-URL download and upload, same behaviour as the splat worker."""
from __future__ import annotations

import os
import time
from pathlib import Path
from urllib.parse import urlparse

import requests

DOWNLOAD_CHUNK = 8 * 1024 * 1024
DOWNLOAD_CONNECT_TIMEOUT = 30
DOWNLOAD_READ_TIMEOUT = int(os.environ.get("DOWNLOAD_READ_TIMEOUT", "120"))
DOWNLOAD_TOTAL_TIMEOUT = int(os.environ.get("DOWNLOAD_TOTAL_TIMEOUT", "3600"))
DOWNLOAD_LOG_INTERVAL = 15
MB = 1024 ** 2


def require_http_url(value, field: str) -> str:
    if not value:
        raise ValueError(f"Missing required field: {field}")
    if urlparse(str(value)).scheme not in ("http", "https"):
        raise ValueError(f"{field} must be an http(s) URL")
    return str(value)


def download(url: str, dest: Path, headers=None):
    started = last_log = time.time()
    total = 0
    with requests.get(url, headers=headers or {}, stream=True,
                      timeout=(DOWNLOAD_CONNECT_TIMEOUT, DOWNLOAD_READ_TIMEOUT)) as response:
        response.raise_for_status()
        expected = response.headers.get("Content-Length")
        if expected:
            print(f"  archive size: {int(expected) / MB:.1f} MB", flush=True)
        with dest.open("wb") as f:
            for chunk in response.iter_content(chunk_size=DOWNLOAD_CHUNK):
                if chunk:
                    f.write(chunk)
                    total += len(chunk)
                now = time.time()
                if now - started > DOWNLOAD_TOTAL_TIMEOUT:
                    raise TimeoutError(f"source download exceeded {DOWNLOAD_TOTAL_TIMEOUT}s "
                                       f"({total / MB:.1f} MB received)")
                if now - last_log >= DOWNLOAD_LOG_INTERVAL:
                    print(f"  downloading… {total / MB:.1f} MB", flush=True)
                    last_log = now
    print(f"download complete ({total / MB:.1f} MB) in {time.time() - started:.0f}s", flush=True)
    return total


def upload(url: str, path: Path, method="PUT", headers=None, content_type=None):
    method = method.upper()
    if method not in ("PUT", "POST"):
        raise ValueError("output_method must be PUT or POST")
    headers = dict(headers or {})
    if content_type and not any(k.lower() == "content-type" for k in headers):
        headers["Content-Type"] = content_type
    with path.open("rb") as f:
        response = requests.request(method, url, headers=headers, data=f, timeout=(30, 1800))
    response.raise_for_status()


def refresh_upload_urls(refresh_url):
    """Re-sign upload links right before uploading (signed links expire).

    Returns the app's JSON dict, or None to keep the links from the payload.
    """
    if not refresh_url:
        return None
    for attempt in range(1, 4):
        try:
            response = requests.post(refresh_url, timeout=(15, 60))
            if response.status_code == 200:
                data = response.json()
                if isinstance(data, dict):
                    print("refreshed signed upload links", flush=True)
                    return data
                return None
            print(f"upload link refresh attempt {attempt}: HTTP {response.status_code}", flush=True)
            if 400 <= response.status_code < 500:
                return None
        except Exception as exc:  # noqa: BLE001 - fall back to original links
            print(f"upload link refresh attempt {attempt} failed: {exc}", flush=True)
        time.sleep(2 * attempt)
    return None
