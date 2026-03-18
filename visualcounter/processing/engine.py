from __future__ import annotations

import logging
import threading
import time
from collections import deque
from dataclasses import dataclass

import cv2
import numpy as np

from visualcounter.config import CameraSettings, Point
from visualcounter.detectors.base import Detector
from visualcounter.models import Detection, Snapshot
from visualcounter.roi import count_in_polygon, count_in_roi, roi_polygon, transform_roi_for_source_crop
from visualcounter.smoothing.base import Smoother

logger = logging.getLogger(__name__)


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
        self._logged_roi_shapes: set[tuple[int, int]] = set()
        self._logged_crop_shapes: set[tuple[int, int]] = set()
        self._logged_clipped_roi_crops: set[tuple[float, float, float, float]] = set()

    def start(self) -> None:
        logger.info("Starting camera worker '%s' for source %s", self.camera_name, self.settings.source_url)
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

        processed_roi = self._roi_for_processing(roi)
        raw_count = 0 if len(processed_roi) < 3 else count_in_roi(snapshot.detections, processed_roi, snapshot.frame_shape)

        smoothed_value: float | None = None
        smoothing_type: str | None = None
        if self._smoother is not None:
            samples = self._build_count_samples(history, processed_roi)
            smoothed_value = self._smoother.smooth(samples, snapshot.timestamp)
            smoothing_type = type(self._smoother).__name__

        return CountResult(
            camera=self.camera_name,
            roi_name=roi_name,
            roi=processed_roi,
            count=raw_count,
            smoothed_count=smoothed_value,
            smoothing_type=smoothing_type,
            timestamp=snapshot.timestamp,
            sequence=snapshot.sequence,
        )

    def _build_count_samples(self, snapshots: list[Snapshot], roi: list[Point]) -> list[tuple[float, float]]:
        if len(roi) < 3:
            return [(snapshot.timestamp, 0.0) for snapshot in snapshots]

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
        should_log = False
        with self._condition:
            should_log = self._last_error != message
            self._last_error = message
        if should_log:
            logger.warning("Camera '%s': %s", self.camera_name, message)

    def _clear_error(self) -> None:
        cleared_error: str | None = None
        with self._condition:
            cleared_error = self._last_error
            self._last_error = None
        if cleared_error is not None:
            logger.info("Camera '%s' recovered after error: %s", self.camera_name, cleared_error)

    def _frame_rois(self, frame_shape: tuple[int, int]) -> dict[str, np.ndarray]:
        rois: dict[str, np.ndarray] = {}
        for name, points in self.settings.rois.items():
            transformed = self._roi_for_processing(points)
            if len(transformed) < 3:
                continue
            rois[name] = roi_polygon(transformed, frame_shape)
        return rois

    def _preview_rois(self, frame_shape: tuple[int, int]) -> dict[str, np.ndarray]:
        rois: dict[str, np.ndarray] = {}
        for name, points in self.settings.rois.items():
            if len(points) < 3:
                continue
            rois[name] = roi_polygon(points, frame_shape)
        return rois

    def _roi_for_processing(self, roi: list[Point]) -> list[Point]:
        processed_roi = transform_roi_for_source_crop(roi, self.settings.processing.source_crop)
        crop = self.settings.processing.source_crop
        if crop is not None and len(processed_roi) < 3 and crop not in self._logged_clipped_roi_crops:
            self._logged_clipped_roi_crops.add(crop)
            logger.warning(
                "Camera '%s': ROI was clipped out by source_crop=%s and is no longer usable",
                self.camera_name,
                crop,
            )
        return processed_roi

    def _offset_detections(
        self,
        detections: list[Detection],
        offset_x: int,
        offset_y: int,
    ) -> list[Detection]:
        if offset_x == 0 and offset_y == 0:
            return detections

        return [
            Detection(
                x=det.x + offset_x,
                y=det.y + offset_y,
                width=det.width,
                height=det.height,
                confidence=det.confidence,
            )
            for det in detections
        ]

    def _source_crop_rect(self, frame_shape: tuple[int, int]) -> tuple[int, int, int, int] | None:
        source_crop = self.settings.processing.source_crop
        if source_crop is None:
            return None

        frame_h, frame_w = frame_shape
        if frame_h <= 0 or frame_w <= 0:
            return None

        x1_n, y1_n, x2_n, y2_n = source_crop
        x1 = max(0, min(frame_w - 1, int(np.floor(x1_n * frame_w))))
        y1 = max(0, min(frame_h - 1, int(np.floor(y1_n * frame_h))))
        x2 = max(x1 + 1, min(frame_w, int(np.ceil(x2_n * frame_w))))
        y2 = max(y1 + 1, min(frame_h, int(np.ceil(y2_n * frame_h))))
        return (x1, y1, x2, y2)

    def _capture_timeout_params(self) -> list[int]:
        processing = self.settings.processing
        params: list[int] = []

        open_timeout = processing.ffmpeg_open_timeout_ms
        if open_timeout is not None and hasattr(cv2, "CAP_PROP_OPEN_TIMEOUT_MSEC"):
            params.extend([int(cv2.CAP_PROP_OPEN_TIMEOUT_MSEC), int(open_timeout)])

        read_timeout = processing.ffmpeg_read_timeout_ms
        if read_timeout is not None and hasattr(cv2, "CAP_PROP_READ_TIMEOUT_MSEC"):
            params.extend([int(cv2.CAP_PROP_READ_TIMEOUT_MSEC), int(read_timeout)])

        return params

    def _open_capture(self) -> cv2.VideoCapture:
        params = self._capture_timeout_params()
        cap: cv2.VideoCapture | None = None

        if params and hasattr(cv2, "CAP_FFMPEG"):
            try:
                cap = cv2.VideoCapture(self.settings.source_url, int(cv2.CAP_FFMPEG), params)
            except (TypeError, cv2.error):
                cap = None

        if cap is None or not cap.isOpened():
            if cap is not None:
                cap.release()
            cap = cv2.VideoCapture(self.settings.source_url)

        # Hint to reduce upstream buffering where backend supports it.
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        return cap

    def _reconnect_delay(self) -> float:
        return self.settings.processing.reconnect_delay_seconds

    def _run(self) -> None:
        processing = self.settings.processing

        frame_index = 0
        last_processed_ts = 0.0
        last_infer_ts = 0.0
        latest_preview_detections: list[Detection] = []

        frame_queue_size = processing.latest_frame_queue_size
        preview_pixel_rois: dict[str, np.ndarray] | None = None
        preview_roi_shape: tuple[int, int] | None = None
        source_crop_shape: tuple[int, int] | None = None
        source_crop_rect: tuple[int, int, int, int] | None = None

        try:
            while not self._stop_event.is_set():
                cap = self._open_capture()
                if not cap.isOpened():
                    self._set_error(f"Failed to open source: {self.settings.source_url}")
                    cap.release()
                    if self._stop_event.wait(self._reconnect_delay()):
                        break
                    continue

                logger.info("Camera '%s': source opened successfully", self.camera_name)

                frame_queue: deque[np.ndarray] = deque(maxlen=frame_queue_size) if frame_queue_size > 0 else deque()
                frame_queue_lock = threading.Lock()
                frame_ready = threading.Condition(frame_queue_lock)
                reconnect_requested = threading.Event()
                read_failures = 0
                capture_thread: threading.Thread | None = None

                def capture_frames_to_queue() -> None:
                    nonlocal read_failures

                    while not self._stop_event.is_set() and not reconnect_requested.is_set():
                        ok, queued_frame = cap.read()
                        if not ok:
                            read_failures += 1
                            if read_failures >= processing.read_failures_before_reconnect:
                                self._set_error("Failed to read frame from source")
                                reconnect_requested.set()
                                with frame_ready:
                                    frame_ready.notify_all()
                                return
                            time.sleep(0.05)
                            continue

                        read_failures = 0
                        with frame_ready:
                            frame_queue.append(queued_frame)
                            frame_ready.notify_all()

                if frame_queue_size > 0:
                    capture_thread = threading.Thread(
                        target=capture_frames_to_queue,
                        name=f"camera-capture-{self.camera_name}",
                        daemon=True,
                    )
                    capture_thread.start()

                try:
                    while not self._stop_event.is_set():
                        if frame_queue_size > 0:
                            with frame_ready:
                                if not frame_queue and not reconnect_requested.is_set():
                                    frame_ready.wait(timeout=0.5)
                                if frame_queue:
                                    frame = frame_queue.popleft()
                                elif reconnect_requested.is_set():
                                    break
                                else:
                                    continue
                        else:
                            ok, frame = cap.read()
                            if not ok:
                                read_failures += 1
                                if read_failures >= processing.read_failures_before_reconnect:
                                    self._set_error("Failed to read frame from source")
                                    break
                                time.sleep(0.05)
                                continue
                            read_failures = 0

                        if processing.scale != 1.0:
                            frame = cv2.resize(
                                frame,
                                None,
                                fx=processing.scale,
                                fy=processing.scale,
                                interpolation=cv2.INTER_AREA,
                            )

                        preview_frame = frame

                        if preview_pixel_rois is None or preview_roi_shape != preview_frame.shape[:2]:
                            preview_roi_shape = preview_frame.shape[:2]
                            preview_pixel_rois = self._preview_rois(preview_roi_shape)

                        if source_crop_rect is None or source_crop_shape != preview_frame.shape[:2]:
                            source_crop_shape = preview_frame.shape[:2]
                            source_crop_rect = self._source_crop_rect(source_crop_shape)
                            if source_crop_rect is not None and source_crop_shape not in self._logged_crop_shapes:
                                self._logged_crop_shapes.add(source_crop_shape)
                                logger.info(
                                    "Camera '%s': source_crop=%s resolved to pixels=%s on frame_shape=%s",
                                    self.camera_name,
                                    self.settings.processing.source_crop,
                                    source_crop_rect,
                                    source_crop_shape,
                                )

                        processing_frame = preview_frame
                        crop_offset_x = 0
                        crop_offset_y = 0
                        if source_crop_rect is not None:
                            sx1, sy1, sx2, sy2 = source_crop_rect
                            processing_frame = preview_frame[sy1:sy2, sx1:sx2]
                            crop_offset_x = sx1
                            crop_offset_y = sy1
                            if processing_frame.size == 0:
                                self._set_error(f"Source crop produced an empty frame: {source_crop_rect}")
                                time.sleep(0.05)
                                continue

                        processing_shape = processing_frame.shape[:2]
                        if processing_shape not in self._logged_roi_shapes:
                            self._logged_roi_shapes.add(processing_shape)
                            logger.info(
                                "Camera '%s': processing frame_shape=%s, preview frame_shape=%s",
                                self.camera_name,
                                processing_shape,
                                preview_frame.shape[:2],
                            )

                        frame_index += 1
                        if processing.every_n_frames > 1 and frame_index % processing.every_n_frames != 0:
                            if self._preview_enabled and self._render_preview(
                                preview_frame,
                                latest_preview_detections,
                                preview_pixel_rois,
                                source_crop_rect,
                            ):
                                return
                            continue

                        now = time.time()

                        if processing.max_processed_fps and processing.max_processed_fps > 0:
                            min_gap = 1.0 / processing.max_processed_fps
                            if (now - last_processed_ts) < min_gap:
                                if self._preview_enabled and self._render_preview(
                                    preview_frame,
                                    latest_preview_detections,
                                    preview_pixel_rois,
                                    source_crop_rect,
                                ):
                                    return
                                continue

                        if processing.infer_every_seconds > 0 and (now - last_infer_ts) < processing.infer_every_seconds:
                            if self._preview_enabled and self._render_preview(
                                preview_frame,
                                latest_preview_detections,
                                preview_pixel_rois,
                                source_crop_rect,
                            ):
                                return
                            continue

                        try:
                            detections = self._detector.infer(processing_frame)
                        except Exception as exc:  # runtime model errors should not kill the worker
                            self._set_error(f"Inference failed: {exc}")
                            time.sleep(0.05)
                            continue

                        latest_preview_detections = self._offset_detections(detections, crop_offset_x, crop_offset_y)
                        self._clear_error()
                        self._record_snapshot(now, processing_frame.shape[:2], detections)
                        if self._preview_enabled and self._render_preview(
                            preview_frame,
                            latest_preview_detections,
                            preview_pixel_rois,
                            source_crop_rect,
                        ):
                            return
                        last_processed_ts = now
                        last_infer_ts = now
                finally:
                    reconnect_requested.set()
                    with frame_ready:
                        frame_ready.notify_all()
                    if capture_thread is not None:
                        capture_thread.join(timeout=1.0)
                    cap.release()

                if not self._stop_event.is_set():
                    logger.info(
                        "Camera '%s': reconnecting to source in %.1f seconds",
                        self.camera_name,
                        self._reconnect_delay(),
                    )
                    if self._stop_event.wait(self._reconnect_delay()):
                        break
        finally:
            self._stop_event.set()
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
        source_crop_rect: tuple[int, int, int, int] | None = None,
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

        if source_crop_rect is not None:
            sx1, sy1, sx2, sy2 = source_crop_rect
            cv2.rectangle(overlay, (sx1, sy1), (sx2, sy2), (255, 255, 0), 1)

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
