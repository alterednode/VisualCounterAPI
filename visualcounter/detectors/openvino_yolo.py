from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from visualcounter.config import DetectorSettings
from visualcounter.detectors.base import Detector, DetectorFactory
from visualcounter.models import Detection


class OpenVinoYoloDetector(Detector):
    def __init__(self, settings: DetectorSettings) -> None:
        try:
            import openvino as ov
        except ImportError as exc:
            raise RuntimeError("OpenVINO is required for detector type 'openvino_yolo'.") from exc

        model_path = Path(settings.model_path)
        if not model_path.exists():
            raise FileNotFoundError(f"Model XML not found: {model_path}")
        bin_path = model_path.with_suffix(".bin")
        if not bin_path.exists():
            raise FileNotFoundError(f"Model BIN not found: {bin_path}")

        self._settings = settings

        core = ov.Core()
        model = core.read_model(str(model_path))
        self._compiled = core.compile_model(model, settings.device)
        self._input_layer = self._compiled.input(0)
        self._output_layer = self._compiled.output(0)

    def infer(self, frame: np.ndarray) -> list[Detection]:
        input_tensor, ratio, dwdh = self._preprocess(frame)
        result = self._compiled({self._input_layer: input_tensor})
        preds = result[self._output_layer]
        return self._postprocess(preds, ratio, dwdh, frame.shape)

    def _letterbox(self, img: np.ndarray) -> tuple[np.ndarray, float, tuple[int, int]]:
        new_shape = self._settings.model_size
        h, w = img.shape[:2]
        ratio = min(new_shape / h, new_shape / w)
        new_unpad = (int(round(w * ratio)), int(round(h * ratio)))
        dw = new_shape - new_unpad[0]
        dh = new_shape - new_unpad[1]
        left = dw // 2
        right = dw - left
        top = dh // 2
        bottom = dh - top

        resized = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
        padded = cv2.copyMakeBorder(
            resized,
            top,
            bottom,
            left,
            right,
            cv2.BORDER_CONSTANT,
            value=(114, 114, 114),
        )
        return padded, ratio, (left, top)

    def _preprocess(self, frame: np.ndarray) -> tuple[np.ndarray, float, tuple[int, int]]:
        image, ratio, dwdh = self._letterbox(frame)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = image.astype(np.float32) / 255.0
        image = np.transpose(image, (2, 0, 1))[None, ...]
        return image, ratio, dwdh

    def _postprocess(
        self,
        preds: np.ndarray,
        ratio: float,
        dwdh: tuple[int, int],
        frame_shape: tuple[int, int],
    ) -> list[Detection]:
        settings = self._settings

        if preds.ndim == 3:
            preds = preds[0]
        if preds.shape[0] < preds.shape[1]:
            preds = preds.T

        boxes = preds[:, :4]
        scores = preds[:, 4:]

        class_ids = np.argmax(scores, axis=1)
        conf = scores[np.arange(scores.shape[0]), class_ids]

        keep = (conf >= settings.conf_threshold) & (class_ids == settings.person_class_id)
        boxes = boxes[keep]
        conf = conf[keep]

        if boxes.size == 0:
            return []

        boxes_xyxy = np.zeros_like(boxes)
        boxes_xyxy[:, 0] = boxes[:, 0] - boxes[:, 2] / 2
        boxes_xyxy[:, 1] = boxes[:, 1] - boxes[:, 3] / 2
        boxes_xyxy[:, 2] = boxes[:, 0] + boxes[:, 2] / 2
        boxes_xyxy[:, 3] = boxes[:, 1] + boxes[:, 3] / 2

        dw, dh = dwdh
        boxes_xyxy[:, [0, 2]] -= dw
        boxes_xyxy[:, [1, 3]] -= dh
        boxes_xyxy /= ratio

        h, w = frame_shape[:2]
        boxes_xyxy[:, [0, 2]] = np.clip(boxes_xyxy[:, [0, 2]], 0, w - 1)
        boxes_xyxy[:, [1, 3]] = np.clip(boxes_xyxy[:, [1, 3]], 0, h - 1)

        boxes_xywh: list[list[int]] = []
        for x1, y1, x2, y2 in boxes_xyxy:
            boxes_xywh.append([int(x1), int(y1), int(x2 - x1), int(y2 - y1)])

        idxs = cv2.dnn.NMSBoxes(
            boxes_xywh,
            conf.tolist(),
            settings.conf_threshold,
            settings.nms_threshold,
        )
        if len(idxs) == 0:
            return []

        idx_values: list[int] = []
        if isinstance(idxs, np.ndarray):
            idx_values = [int(i) for i in idxs.flatten()]
        else:
            idx_values = [int(i[0]) if isinstance(i, (list, tuple)) else int(i) for i in idxs]

        detections: list[Detection] = []
        for i in idx_values:
            x, y, bw, bh = boxes_xywh[i]
            detections.append(
                Detection(
                    x=x,
                    y=y,
                    width=bw,
                    height=bh,
                    confidence=float(conf[i]),
                )
            )
        return detections


class OpenVinoYoloFactory(DetectorFactory):
    def create(self, settings: DetectorSettings) -> Detector:
        return OpenVinoYoloDetector(settings)
