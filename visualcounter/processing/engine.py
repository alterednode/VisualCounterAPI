from __future__ import annotations

import threading
import time
from collections import deque
from dataclasses import dataclass

import cv2
import numpy as np

from visualcounter.config import CameraSettings, Point
from visualcounter.detectors.base import Detector
from visualcounter.models import Detection, Snapshot
from visualcounter.roi import count_in_polygon, count_in_roi, roi_polygon
from visualcounter.smoothing.base import Smoother


@dataclass(frozen=True)
class CountResult:
    camera: str
    roi_name: str | None
    roi: list[Point]
    count: int
    smoothed_count: float | None
    smoothing_type: str | None
    timestamp: float
    sequence: int


class CameraWorker:
    _gui_lock = threading.Lock()

    def __init__(
        self,
        camera_name: str,
        settings: CameraSettings,
        detector: Detector,
        smoother: Smoother | None,
    ) -> None:
        self.camera_name = camera_name
        self.settings = settings
        self._detector = detector
        self._smoother = smoother

        self._thread = threading.Thread(target=self._run, name=f"camera-{camera_name}", daemon=True)
        self._stop_event = threading.Event()

        self._condition = threading.Condition()
        self._latest_snapshot: Snapshot | None = None
        self._history: deque[Snapshot] = deque()
        self._sequence = 0
        self._last_error: str | None = None
        self._preview_enabled = bool(settings.processing.show_preview)
        self._preview_window_name = f"{camera_name}-preview"

    def start(self) -> None:
        self._thread.start()

    def stop(self) -> None:
        self._stop_event.set()
        self._thread.join(timeout=5.0)

    def get_rois(self) -> dict[str, list[Point]]:
        return {name: list(points) for name, points in self.settings.rois.items()}

    def get_last_error(self) -> str | None:
        with self._condition:
            return self._last_error

    def wait_for_update(self, last_sequence: int, timeout_seconds: float) -> Snapshot | None:
        with self._condition:
            if self._latest_snapshot is not None and self._latest_snapshot.sequence > last_sequence:
                return self._latest_snapshot
            self._condition.wait(timeout=timeout_seconds)
            if self._latest_snapshot is not None and self._latest_snapshot.sequence > last_sequence:
                return self._latest_snapshot
            return None

    def get_count(self, roi: list[Point], roi_name: str | None = None) -> CountResult:
        with self._condition:
            snapshot = self._latest_snapshot
            history = list(self._history)

        if snapshot is None:
            raise RuntimeError(f"No frames have been processed for camera '{self.camera_name}' yet")

        raw_count = count_in_roi(snapshot.detections, roi, snapshot.frame_shape)

        smoothed_value: float | None = None
        smoothing_type: str | None = None
        if self._smoother is not None:
            samples = self._build_count_samples(history, roi)
            smoothed_value = self._smoother.smooth(samples, snapshot.timestamp)
            smoothing_type = type(self._smoother).__name__

        return CountResult(
            camera=self.camera_name,
            roi_name=roi_name,
            roi=roi,
            count=raw_count,
            smoothed_count=smoothed_value,
            smoothing_type=smoothing_type,
            timestamp=snapshot.timestamp,
            sequence=snapshot.sequence,
        )

    def _build_count_samples(self, snapshots: list[Snapshot], roi: list[Point]) -> list[tuple[float, float]]:
        samples: list[tuple[float, float]] = []
        for snapshot in snapshots:
            samples.append(
                (
                    snapshot.timestamp,
                    float(count_in_roi(snapshot.detections, roi, snapshot.frame_shape)),
                )
            )
        return samples

    def _record_snapshot(self, timestamp: float, frame_shape: tuple[int, int], detections: list[Detection]) -> None:
        with self._condition:
            self._sequence += 1
            snapshot = Snapshot(
                sequence=self._sequence,
                timestamp=timestamp,
                frame_shape=frame_shape,
                detections=detections,
            )
            self._latest_snapshot = snapshot
            self._history.append(snapshot)
            self._prune_history(timestamp)
            self._condition.notify_all()

    def _prune_history(self, now: float) -> None:
        retention = 2.0
        if self._smoother is not None:
            retention = max(retention, self._smoother.retention_seconds + 1.0)

        min_timestamp = now - retention
        while len(self._history) > 1 and self._history[0].timestamp < min_timestamp:
            self._history.popleft()

    def _set_error(self, message: str) -> None:
        with self._condition:
            self._last_error = message

    def _clear_error(self) -> None:
        with self._condition:
            self._last_error = None

    def _frame_rois(self, frame_shape: tuple[int, int]) -> dict[str, np.ndarray]:
        rois: dict[str, np.ndarray] = {}
        for name, points in self.settings.rois.items():
            rois[name] = roi_polygon(points, frame_shape)
        return rois

    def _crop_rect(
        self,
        frame_shape: tuple[int, int],
        pixel_rois: dict[str, np.ndarray],
    ) -> tuple[int, int, int, int] | None:
        if not self.settings.processing.crop_to_roi:
            return None
        if not pixel_rois:
            return None

        all_points = np.concatenate(list(pixel_rois.values()), axis=0)
        x, y, w, h = cv2.boundingRect(all_points)
        pad = self.settings.processing.crop_pad

        x1 = max(0, x - pad)
        y1 = max(0, y - pad)
        x2 = min(frame_shape[1], x + w + pad)
        y2 = min(frame_shape[0], y + h + pad)

        if x2 <= x1 or y2 <= y1:
            return None
        return (x1, y1, x2, y2)

    def _run(self) -> None:
        processing = self.settings.processing

        frame_index = 0
        last_processed_ts = 0.0
        last_infer_ts = 0.0
        latest_detections: list[Detection] = []

        cap = cv2.VideoCapture(self.settings.source_url)
        if not cap.isOpened():
            self._set_error(f"Failed to open source: {self.settings.source_url}")
            return

        pixel_rois: dict[str, np.ndarray] | None = None
        roi_shape: tuple[int, int] | None = None
        crop_rect: tuple[int, int, int, int] | None = None

        try:
            while not self._stop_event.is_set():
                ok, frame = cap.read()
                if not ok:
                    self._set_error("Failed to read frame from source")
                    time.sleep(0.05)
                    continue

                if processing.scale != 1.0:
                    frame = cv2.resize(
                        frame,
                        None,
                        fx=processing.scale,
                        fy=processing.scale,
                        interpolation=cv2.INTER_AREA,
                    )

                if pixel_rois is None or roi_shape != frame.shape[:2]:
                    roi_shape = frame.shape[:2]
                    pixel_rois = self._frame_rois(roi_shape)
                    crop_rect = self._crop_rect(roi_shape, pixel_rois)

                frame_index += 1
                if processing.every_n_frames > 1 and frame_index % processing.every_n_frames != 0:
                    if self._preview_enabled and self._render_preview(frame, latest_detections, pixel_rois):
                        break
                    continue

                now = time.time()

                if processing.max_processed_fps and processing.max_processed_fps > 0:
                    min_gap = 1.0 / processing.max_processed_fps
                    if (now - last_processed_ts) < min_gap:
                        if self._preview_enabled and self._render_preview(frame, latest_detections, pixel_rois):
                            break
                        continue

                if processing.infer_every_seconds > 0 and (now - last_infer_ts) < processing.infer_every_seconds:
                    if self._preview_enabled and self._render_preview(frame, latest_detections, pixel_rois):
                        break
                    continue

                infer_frame = frame
                offset_x = 0
                offset_y = 0

                if crop_rect is not None:
                    x1, y1, x2, y2 = crop_rect
                    infer_frame = frame[y1:y2, x1:x2]
                    offset_x = x1
                    offset_y = y1

                try:
                    detections = self._detector.infer(infer_frame)
                except Exception as exc:  # runtime model errors should not kill the worker
                    self._set_error(f"Inference failed: {exc}")
                    time.sleep(0.05)
                    continue

                if offset_x or offset_y:
                    detections = [
                        Detection(
                            x=det.x + offset_x,
                            y=det.y + offset_y,
                            width=det.width,
                            height=det.height,
                            confidence=det.confidence,
                        )
                        for det in detections
                    ]

                latest_detections = detections
                self._clear_error()
                self._record_snapshot(now, frame.shape[:2], detections)
                if self._preview_enabled and self._render_preview(frame, latest_detections, pixel_rois):
                    break
                last_processed_ts = now
                last_infer_ts = now
        finally:
            cap.release()
            if self._preview_enabled:
                try:
                    with self._gui_lock:
                        cv2.destroyWindow(self._preview_window_name)
                except cv2.error:
                    pass

    def _render_preview(
        self,
        frame: np.ndarray,
        detections: list[Detection],
        pixel_rois: dict[str, np.ndarray] | None,
    ) -> bool:
        overlay = frame.copy()

        for det in detections:
            x = det.x
            y = det.y
            w = det.width
            h = det.height
            cv2.rectangle(overlay, (x, y), (x + w, y + h), (0, 255, 0), 1)
            cv2.putText(
                overlay,
                f"{det.confidence:.2f}",
                (x, max(20, y - 5)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (0, 255, 0),
                1,
                cv2.LINE_AA,
            )

        roi_map = pixel_rois or {}
        for roi_name, polygon in roi_map.items():
            cv2.polylines(overlay, [polygon], True, (0, 255, 255), 1)
            count = count_in_polygon(detections, polygon)
            label_anchor = tuple(int(v) for v in polygon[0])
            cv2.putText(
                overlay,
                f"{roi_name}: {count}",
                label_anchor,
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (0, 255, 255),
                2,
                cv2.LINE_AA,
            )

        cv2.putText(
            overlay,
            f"{self.camera_name} detections: {len(detections)}",
            (10, 24),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

        try:
            with self._gui_lock:
                cv2.imshow(self._preview_window_name, overlay)
                key = cv2.waitKey(1) & 0xFF
        except cv2.error as exc:
            self._preview_enabled = False
            self._set_error(f"Preview disabled: {exc}")
            return False

        if key == ord("q"):
            self._stop_event.set()
            return True

        return False
