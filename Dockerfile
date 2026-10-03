# CPU-only image: back-projection, visibility and blending are numpy/OpenCV
# work, so this runs on RunPod CPU endpoints (no CUDA, no GPU billing).
FROM python:3.11-slim-bookworm

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    OMP_NUM_THREADS=0

RUN apt-get update && apt-get install -y --no-install-recommends ca-certificates \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt /tmp/requirements.txt
RUN python3 -m pip install -r /tmp/requirements.txt \
    && python3 -c "import cv2, numpy, tifffile, laspy, pyvips; print('deps ok, vips', pyvips.version(0), pyvips.version(1))"

WORKDIR /worker
COPY facade /worker/facade
COPY tools /worker/tools
COPY handler.py /worker/handler.py

CMD ["python3", "-u", "/worker/handler.py"]
