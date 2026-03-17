# VisualCounter API

Configurable multi-camera person counting service with pluggable detector and smoothing modules.

## Features

- YAML configuration with `defaults` and per-camera overrides.
- Per-camera `source_url` is required (global/default source URL is intentionally disallowed).
- Per-camera ROI presets plus optional ad-hoc ROI queries.
- REST endpoints for count and ROI discovery.
- SSE endpoints for live count updates.
- Modular architecture for detector backends and smoothing algorithms.
- Frame cadence controls: scale, frame stride, max processed FPS, infer interval.

## Legal

Permission from UBCO must be obtained if hosting this project with UBC streams beyond personal use, according to the Terms of Use and Copyright information linked on the [Current Students](https://ok.ubc.ca/current-students) page.

For any other stream, obtain permission from the stream owner as needed.

This project is under the MIT license.

## Quick Start

1. (Optional) Install export tooling and export a YOLO model to OpenVINO XML/BIN:

```bash
uv sync --extra export
uv run python tools/export_yolo.py --model yolov8s.pt --output-dir models --dynamic --half
```

2. Edit `config.yaml` for your stream URLs, model path, and ROIs.
3. Run the API:

```bash
uv run visualcounter-api
```

By default, config is loaded from `config.yaml`. Override with:

```bash
VISUALCOUNTER_CONFIG=/path/to/config.yaml uv run visualcounter-api
```

To enable API key auth, set `VISUALCOUNTER_API_KEYS` to a comma-separated list and send one of the keys in the `X-API-Key` header:

```bash
VISUALCOUNTER_API_KEYS=dev-key-1,dev-key-2 uv run visualcounter-api
```

Example request:

```bash
curl -H 'X-API-Key: dev-key-1' http://localhost:8000/timcam_courtyard/count
```

## API

- `GET /{camera_name}/count?roi_name=queue`
- `GET /{camera_name}/count?roi=0.3047,0.4514;0.3516,0.3333;0.5625,0.3333;0.5625,0.4514`
- `GET /{camera_name}/count/stream?roi_name=queue`
- `GET /{camera_name}/count/live?roi_name=queue`
- `GET /{camera_name}/rois`

If no ROI query is provided, the API uses `default_roi` when configured.

## Configuration Shape

See `config.yaml` for a full example.

Top-level keys:

- `defaults`: shared base settings applied to all cameras (except `source_url`).
- `cameras`: per-camera settings and overrides.

Camera settings include:

- `source_url`
- `detector` (`type`, `model_path`, `device`, thresholds, etc.)
- `processing` (`scale`, `every_n_frames`, `max_processed_fps`, `infer_every_seconds`, crop settings, `show_preview`)
- `smoothing` (optional; enabled only when present)
- `rois` (named polygons in normalized coordinates `0..1`; each camera can define multiple ROI presets)
- `default_roi` (optional)

When `processing.show_preview` is `true`, a local OpenCV preview window is shown for that camera with ROI and detection overlays. Press `q` in the preview window to stop that worker.

Cropping behavior:

- `processing.source_crop` (optional) crops the source frame first, before ROI polygon conversion and inference.
- ROI points are still authored in full-source normalized coordinates.
- For source-cropped cameras, ROIs are transformed into the cropped coordinate space.
- If an ROI is partially outside the source crop, it is clipped to the crop bounds.
- If clipping leaves fewer than 3 points, that ROI is treated as empty (count is `0`).

Freshness behavior:

- `processing.latest_frame_queue_size` enables queued capture mode when set above `0`.
- A value of `1` keeps only the freshest frame (lowest staleness, highest frame drop).
- Larger values keep a short queue of recent frames (less drop, potentially more staleness).

## Extensibility

- Add detector backends by implementing `visualcounter/detectors/base.py` and registering in `visualcounter/service.py`.
- Add smoothing algorithms by implementing `visualcounter/smoothing/base.py` and registering in `visualcounter/service.py`.
- Reuse the same processing engine with file inputs by setting `source_url` to a local video path.
