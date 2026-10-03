# Epiloft Facade Worker

RunPod Serverless worker that turns an ODM/OpenSfM run plus a wall picked in
Studio into a **full-resolution, multi-photo facade orthomosaic**: the wall
straightened onto a flat, true-scale image, built from the best raw photos
of each spot, with occluding objects removed and exposure seams blended out.

Same job pattern as `epiloft-opensplat-worker` (signed URLs in, signed URLs
out, nothing large in the RunPod payload), but a separate image and endpoint:
this one is **CPU only**.

## How it works

| Stage | What happens | Code |
|---|---|---|
| 1 Cameras | Poses and lens models from `opensfm/reconstruction.json`, projected with OpenSfM's own conventions (perspective, brown, fisheye, fisheye_opencv, radial, simple_radial). | `facade/cameras.py` |
| 2 Grid | The picked corners become a wall frame: U along the wall, V up, W out toward the cameras. Corners picked "inside out" are flipped so the image always reads left to right from outside. | `facade/geometry.py` |
| 3 Depth | The dense point cloud is rasterised onto the plane, frontmost point per cell. Trim, sills and recessed windows sit at their real depth, and a plane picked off the surface (on the eaves line, say) still lands on the wall. | `facade/depth.py` |
| 4 Choice | Every candidate photo is scored per spot: square-on, close, and near the frame centre wins. A point-cloud z-buffer per photo rejects views blocked by trees, posts and overhangs. The winner map is smoothed so seams follow regions. | `facade/selection.py`, `facade/visibility.py` |
| 5 Blend | Per-photo exposure gains are solved from overlaps, then a Laplacian-pyramid blend hides the seams without ghosting fine detail. Output is built in tiles, so wall size is bounded by disk, not RAM. | `facade/selection.py`, `facade/blend.py`, `facade/pipeline.py` |
| 6 Outputs | Tiled BigTIFF, sidecar JSON, JPEG preview, optional deep-zoom tiles. | `facade/outputs.py` |

## Lens models are clipped to the photo

Polynomial lens models are only fitted inside the photo. The DJI M4E
calibration on the first real job folds back at 53 degrees off-axis and maps
rays at 63.5 degrees onto the image centre, so wall far outside a photo was
"seen" in it. `Camera.valid_angle` finds, per lens and image size, the
largest ray angle that projects monotonically and lands no further than just
past the image corners; anything beyond it never projects.

## Which frame are the camera poses in?

ODM writes `opensfm/reconstruction.json` either in the georeferenced offset
frame (UTM minus `coords.txt`) or in OpenSfM's local topocentric frame, and
archives such as WebODM Lightning's `all.zip` do not always include the
`reconstruction.topocentric.json` marker that tells them apart. Guessing
wrong is not small: grid convergence alone rotates a site in western South
Dakota by about 1.25 degrees.

So the worker tests both against georeferenced geometry whose frame is
certain, in this order:

1. `odm_georeferencing/odm_georeferenced_model.laz` (absolute UTM)
2. `odm_texturing/odm_textured_model_geo.obj` (offset frame, the mesh Studio shows)

It keeps whichever hypothesis fits the reconstruction's sparse points
better (using a uniform 6-million-point sample of the site), then applies a
small translation-only snap. Once the wall is known, a second, fine snap runs
against full-density geometry at that wall: point-to-plane ICP, translation
only, applied just along directions the geometry constrains (a lone flat
wall pins only the through-wall direction; ground and a corner pin all
three). `local_fit_m`, `local_snap_m` and `constrained_axes` are reported. The decision, the fit
of both hypotheses, the rotation and the snap are recorded in
`facade.json` under `wall_to_world.pose_frame`. A fit worse than 0.30 m,
or no geometry to check against, produces a warning.

Dense geometry for depth and occlusion comes from the LAZ, then the
filterpoints PLY (moved with the poses if they were topocentric), then the
textured mesh vertices.

## RunPod input

```json
{
  "input": {
    "project_id": "ascend-plaza",
    "wall_name": "south elevation",
    "source_url": "https://SIGNED-DOWNLOAD-URL (same ODM ZIP the splat worker gets)",
    "wall": {
      "frame": "mesh",
      "corners": {
        "bottom_left":  [x, y, z],
        "bottom_right": [x, y, z],
        "top_left":     [x, y, z]
      },
      "mesh_origin": {"e": 643850.0, "n": 4884555.0, "z": 1.0},
      "view_from": [x, y, z]
    },
    "gsd_mm": "native",
    "ortho_upload_url":   "https://SIGNED-UPLOAD-URL",
    "sidecar_upload_url": "https://OPTIONAL",
    "jpeg_upload_url":    "https://OPTIONAL (recommended: the file people open)",
    "preview_upload_url": "https://OPTIONAL",
    "tiles_upload_url":   "https://OPTIONAL",
    "upload_url_refresh_url": "https://OPTIONAL per-job re-sign route",
    "options": {}
  }
}
```

**`wall.view_from`** (recommended). Where the viewer stood when picking the
wall, in the same frame as the corners: Studio's camera position. Corners
define a plane, not a side; this says which face is the outside. Without it
the worker decides by visibility: per side, how many (photo, wall point)
pairs are in a photo's real field of view and not blocked by the dense
geometry. The decision is in `facade.json` under `diagnostics.side`.

**`wall.frame`.** `"mesh"` means Studio's three.js scene coordinates (Y up).
The worker inverts the splat worker's `splat_to_mesh` contract exactly
(`tests/test_geometry.py` pins it), so it needs the same `mesh_origin` Studio
uses to place the model. `"opensfm"` takes corners already in the
reconstruction frame. Pick the corners on the wall face; the depth stage
tolerates being off the surface by up to `depth_front_m` / `depth_back_m`.

**`gsd_mm`.** `"native"` (default) uses the median resolution of the photos
that won, which is the most detail the capture holds. A number forces a scale;
finer than native just makes a bigger file and raises a warning.

**Refresh route** returns JSON with any of the four `*_upload_url` keys, like
the splat worker's.

### Options

All optional. Unknown keys are rejected so a typo never silently does nothing.

| Key | Default | Notes |
|---|---|---|
| `depth_front_m` / `depth_back_m` | 0.6 / 0.5 | How far in front of / behind the picked plane to look for the real surface. |
| `depth_cell_mm` | auto | Depth grid cell. Auto = 1.5x the point spacing on the wall, 10 to 100 mm. |
| `use_point_cloud` | true | False = flat plane, no occlusion. Only for debugging. |
| `max_incidence_deg` | 65 | Views more oblique than this are not used. |
| `min_standoff_m` | 0.5 | Camera must be at least this far in front of the wall. |
| `max_cameras` | 80 | Photos used per wall. |
| `top_per_cell` | 6 | Photos occlusion-checked per wall spot per round. All candidates are scored on geometry first; only likely winners get the expensive occlusion check. |
| `edge_margin` | 0.02 | Fraction of each frame edge ignored. |
| `blend_levels` | 5 | Pyramid levels. More = wider brightness blending across seams. |
| `gain_sigma_g` | 1.0 | Exposure gain prior. Lower pulls gains toward 1 (OpenCV uses 0.1, which leaves real exposure steps visible). Gains are then rescaled so their overlap-weighted geometric mean is 1, so levelling never darkens the facade. |
| `zbuffer_cells_per_spacing` | 1.5 | Occlusion z-buffer cell size, in point spacings. |
| `zbuffer_downscale` | auto | Force the z-buffer cell size in source pixels. |
| `occlusion_abs_tol_m` / `occlusion_rel_tol` | 0.08 / 0.01 | How far behind the z-buffer a point may sit and still count as visible. |
| `max_output_megapixels` | 600 | Guardrail; the error says which `gsd_mm` would fit. |
| `image_cache_gb` | 8 | Decoded photo cache. |

## Outputs

| File | What it is |
|---|---|
| `facade.jpg` | **The one to open.** Full-resolution JPEG, white where no photo saw the wall. Opens in any viewer, browser or phone. (Reduced only past JPEG's 65,535 px limit, with a warning.) |
| `facade.tif` | Tiled RGBA TIFF, lossless LZW; classic TIFF unless over ~3.5 GB. Transparent where no photo saw that spot unobstructed. DPI tags carry the real scale. For CAD and GIS. |
| `facade.json` | The sidecar: plane in OpenSfM and mesh frames, scale, pixel to wall to world maths, depth stats, photos used with share, distance and gain, warnings, timings, effective options. |
| `facade_preview.jpg` | Long edge up to 4096 px, alpha flattened onto white. |
| `facade_tiles.zip` | Deep Zoom pyramid (`.dzi` + PNG tiles), only when `tiles_upload_url` is set. |

**Why not GeoTIFF:** a GeoTIFF's georeferencing assumes a horizontal map
projection, which a vertical wall does not have. Scale and position live in
the sidecar instead.

**Measuring:** pixels are x right, y down from the top-left corner. Distance
on the wall plane is `pixel distance * image.gsd_m`. `tools/measure.py` does it
in metres and feet-inches:

```bash
python3 tools/measure.py facade.json 412 1880 1038 1880
```

## Local runs

Same pipeline, no RunPod, against an ODM project folder or its ZIP:

```bash
python3 handler.py --local ./odm_project.zip --wall wall.json --out ./out
```

`wall.json` is the payload's `wall` block (or a whole payload). This is the
way to validate a real capture against tape measurements before turning it
on for customers.

## Capture guidance

The worker can only use photos that face the wall. For each elevation:

- Fly an oblique facade pass, camera roughly square to the wall, with
  60 to 70% overlap both ways.
- Standoff sets resolution: with a 20 MP 1-inch sensor, about 1 mm/px at
  4 m and 2.7 mm/px at 10 m.
- Add a few photos from slightly left and right where trees or posts stand in
  front of the wall, so something sees behind them.
- Nadir mapping photos are ignored for facades; they still help the
  reconstruction.

## Memory

Dense geometry is streamed, never loaded whole: LAZ through laspy's chunk
iterator (multi-threaded decompression), binary PLY through a memory map,
OBJ line by line (`facade/cloud.py`). The worker keeps a 6 M point site-wide
sample for the frame check, every point on the wall surface, and a capped
8 M point sample of anything between the wall and the cameras. A full job
against a 40 M point LAZ peaked at **0.8 GB**; peak memory no longer grows
with site size.

## Performance

Measured on 2 CPU cores: 14 photos at 20.9 MP into a 6667 x 3334 (22 MP)
facade at 0.9 mm/px took **22 s** with **1.5 GB** peak memory. Time scales
roughly with output megapixels times photos per spot. A CPU endpoint with
8 vCPU and 16 to 32 GB RAM is a comfortable starting point; container disk
should be at least 3x the source ZIP.

## Tests

```bash
pip install -r requirements.txt pytest
python3 -m pytest -q
```

`tests/synthetic.py` builds a full fake ODM project: a textured wall at an
angle, a post in front of it, and photos ray-traced through a real Brown lens
model with random exposure. Because the true wall texture is known, the tests
check the output pixel for pixel:

- matches ground truth (PSNR > 27 dB) with sub-pixel alignment
- the post is removed, and without occlusion checks it is not (control)
- a plane picked 35 cm off the wall still lands on the wall, and fails without depth (control)
- exposure steps of 0.75x / 1.25x are levelled
- inside-out corners flip instead of mirroring; mesh and OpenSfM frames agree
- a sparse point cloud still blocks the post (regression from 20 MP testing)
- poses in the topocentric frame with no marker file (the Lightning case) are
  detected against the LAZ or the textured mesh and land sub-pixel; a PLY in
  the original frame moves with them; an unverifiable frame is explained
- LAZ point clouds, deep-zoom tiles, the RunPod handler with signed URL upload and refresh

## Before the first customer job

These depend on real ODM output and are not covered by synthetic tests:

1. **Pose frame.** Check `wall_to_world.pose_frame.fit_m` in the sidecar of
   the first real job; it should be a few centimetres. If a well-picked wall
   has a depth `offset_median_m` far from 0, look here first.
2. **Mesh origin.** Confirm Studio's `mesh_origin` matches the one the splat
   alignment uses; a mismatch shifts the wall in the photos.
3. **Tape check.** Run one wall locally and compare a door width and a window
   height against tape with `tools/measure.py`.
4. **Sparse occluders.** Trees often have sparser points than walls. If foliage
   bleeds onto the wall, raise `zbuffer_cells_per_spacing`.
