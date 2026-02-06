from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field, ValidationError, field_validator, model_validator

Point = tuple[float, float]
RoiMap = dict[str, list[Point]]


class DetectorSettings(BaseModel):
    type: str = "openvino_yolo"
    model_path: str
    device: str = "CPU"
    model_size: int = 640
    person_class_id: int = 0
    conf_threshold: float = 0.15
    nms_threshold: float = 0.5


class ProcessingSettings(BaseModel):
    scale: float = 1.0
    every_n_frames: int = 1
    max_processed_fps: float | None = None
    infer_every_seconds: float = 0.0
    crop_to_roi: bool = False
    crop_pad: int = 40
    show_preview: bool = False


class SmoothingSettings(BaseModel):
    type: str
    params: dict[str, Any] = Field(default_factory=dict)


class CameraSettings(BaseModel):
    source_url: str
    detector: DetectorSettings
    processing: ProcessingSettings = Field(default_factory=ProcessingSettings)
    smoothing: SmoothingSettings | None = None
    rois: RoiMap = Field(default_factory=dict)
    default_roi: str | None = None

    @field_validator("rois", mode="before")
    @classmethod
    def validate_rois(cls, value: Any) -> RoiMap:
        if value is None:
            return {}
        if not isinstance(value, dict):
            raise ValueError("rois must be a mapping of name -> points")

        parsed: RoiMap = {}
        for roi_name, pts in value.items():
            if not isinstance(pts, list) or len(pts) < 3:
                raise ValueError(f"roi '{roi_name}' must define at least 3 points")
            parsed_pts: list[Point] = []
            for point in pts:
                if not isinstance(point, (list, tuple)) or len(point) != 2:
                    raise ValueError(f"roi '{roi_name}' has invalid point {point!r}")
                x, y = point
                x_f = float(x)
                y_f = float(y)
                if not (0.0 <= x_f <= 1.0 and 0.0 <= y_f <= 1.0):
                    raise ValueError(
                        f"roi '{roi_name}' point {(x, y)!r} must be normalized between 0 and 1"
                    )
                parsed_pts.append((x_f, y_f))
            parsed[roi_name] = parsed_pts
        return parsed

    @model_validator(mode="after")
    def validate_default_roi(self) -> "CameraSettings":
        if self.default_roi and self.default_roi not in self.rois:
            raise ValueError(f"default_roi '{self.default_roi}' must exist in rois")
        return self


class AppConfig(BaseModel):
    cameras: dict[str, CameraSettings]


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = deepcopy(base)
    for key, value in override.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = deepcopy(value)
    return merged


def load_config(path: str | Path) -> AppConfig:
    config_path = Path(path)
    data = yaml.safe_load(config_path.read_text())
    if not isinstance(data, dict):
        raise ValueError("Config root must be a mapping")

    defaults = data.get("defaults") or {}
    if not isinstance(defaults, dict):
        raise ValueError("defaults must be a mapping")
    if "source_url" in defaults:
        raise ValueError("defaults.source_url is not allowed; each camera must define its own source_url")

    cameras = data.get("cameras")
    if not isinstance(cameras, dict) or not cameras:
        raise ValueError("cameras must be a non-empty mapping")

    resolved_cameras: dict[str, CameraSettings] = {}
    for camera_name, camera_data in cameras.items():
        if not isinstance(camera_data, dict):
            raise ValueError(f"camera '{camera_name}' configuration must be a mapping")
        if "source_url" not in camera_data:
            raise ValueError(f"camera '{camera_name}' must define source_url explicitly")
        merged = _deep_merge(defaults, camera_data)
        try:
            resolved_cameras[camera_name] = CameraSettings.model_validate(merged)
        except ValidationError as exc:
            raise ValueError(f"Invalid configuration for camera '{camera_name}': {exc}") from exc

    return AppConfig(cameras=resolved_cameras)
