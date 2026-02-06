from __future__ import annotations

from dataclasses import dataclass

from visualcounter.config import AppConfig, Point
from visualcounter.detectors.openvino_yolo import OpenVinoYoloFactory
from visualcounter.detectors.registry import DetectorRegistry
from visualcounter.processing.engine import CameraWorker, CountResult
from visualcounter.roi import parse_roi_string
from visualcounter.smoothing.none import NoSmoothingFactory
from visualcounter.smoothing.registry import SmootherRegistry
from visualcounter.smoothing.time_weighted import TimeWeightedAverageFactory


@dataclass(frozen=True)
class ResolvedRoi:
    name: str | None
    points: list[Point]


class VisualCounterService:
    def __init__(self, config: AppConfig) -> None:
        self._config = config

        detector_registry = DetectorRegistry()
        detector_registry.register("openvino_yolo", OpenVinoYoloFactory())

        smoother_registry = SmootherRegistry()
        smoother_registry.register("none", NoSmoothingFactory())
        smoother_registry.register("time_weighted_average", TimeWeightedAverageFactory())

        self._workers: dict[str, CameraWorker] = {}
        for camera_name, camera_settings in config.cameras.items():
            detector = detector_registry.create(camera_settings.detector)
            smoother = smoother_registry.create(camera_settings.smoothing)
            self._workers[camera_name] = CameraWorker(
                camera_name=camera_name,
                settings=camera_settings,
                detector=detector,
                smoother=smoother,
            )

    def start(self) -> None:
        for worker in self._workers.values():
            worker.start()

    def stop(self) -> None:
        for worker in self._workers.values():
            worker.stop()

    def camera_names(self) -> list[str]:
        return sorted(self._workers.keys())

    def worker(self, camera_name: str) -> CameraWorker:
        try:
            return self._workers[camera_name]
        except KeyError as exc:
            raise KeyError(f"Unknown camera '{camera_name}'") from exc

    def get_rois(self, camera_name: str) -> dict[str, list[Point]]:
        return self.worker(camera_name).get_rois()

    def resolve_roi(self, camera_name: str, roi_name: str | None, roi: str | None) -> ResolvedRoi:
        rois = self.get_rois(camera_name)
        camera_settings = self._config.cameras[camera_name]

        if roi and roi_name:
            raise ValueError("Provide either 'roi_name' or 'roi', not both")

        if roi_name:
            points = rois.get(roi_name)
            if points is None:
                known = ", ".join(sorted(rois.keys())) or "<none>"
                raise ValueError(f"Unknown roi_name '{roi_name}'. Available: {known}")
            return ResolvedRoi(name=roi_name, points=points)

        if roi:
            points = parse_roi_string(roi)
            return ResolvedRoi(name=None, points=points)

        if camera_settings.default_roi:
            return ResolvedRoi(name=camera_settings.default_roi, points=rois[camera_settings.default_roi])

        if len(rois) == 1:
            name = next(iter(rois))
            return ResolvedRoi(name=name, points=rois[name])

        if not rois:
            raise ValueError(
                "No ROI configured for camera and no custom ROI provided. "
                "Define camera rois/default_roi or pass ?roi=0.30,0.45;0.35,0.33;0.56,0.33"
            )

        names = ", ".join(sorted(rois.keys()))
        raise ValueError(f"Multiple ROIs exist. Choose one with roi_name. Available: {names}")

    def get_count(self, camera_name: str, roi_name: str | None, roi: str | None) -> CountResult:
        resolved = self.resolve_roi(camera_name, roi_name=roi_name, roi=roi)
        return self.worker(camera_name).get_count(roi=resolved.points, roi_name=resolved.name)
