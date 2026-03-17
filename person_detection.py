import os
import sys
import time
import cv2
import numpy as np

try:
    import openvino as ov
except ImportError as exc:
    msg = "OpenVINO is required. Install it in your venv, then rerun."
    print(f"ERROR: {msg}", file=sys.stderr)
    raise RuntimeError(msg) from exc

HLS_URL = "https://streamserve.ok.ubc.ca/LiveCams/timcam.stream_720p/playlist.m3u8"

# ROI
ROI = np.array([(390, 325), (450, 240), (720, 240), (720, 325)], dtype=np.int32)
# this assumes a 720p frame. 

# Model/config
MODEL_PATH = "models/yolov8s_openvino_model/yolov8s.xml"
DEVICE = "CPU"
MODEL_SIZE = 640  # YOLOv8 export default
PERSON_CLASS_ID = 0
CONF_THRESHOLD = 0.15
NMS_THRESHOLD = 0.5

# Processing config
SCALE = 1.0  # keep 720p; set <1.0 for speed (ROI auto-scales)
FRAME_STRIDE = 2  # process every frame
INFER_EVERY_SEC = 0  # set to 0 to infer on every processed frame
SHOW_PREVIEW = True
CROP_TO_ROI = True
CROP_PAD = 40
SMOOTH_SECONDS = 10.0


def scale_points(pts: np.ndarray, scale: float) -> np.ndarray:
    if scale == 1.0:
        return pts
    return np.round(pts.astype(np.float32) * scale).astype(np.int32)


def ensure_model_files(xml_path: str) -> None:
    if not os.path.exists(xml_path):
        raise FileNotFoundError(f"Model XML not found: {xml_path}")
    bin_path = os.path.splitext(xml_path)[0] + ".bin"
    if not os.path.exists(bin_path):
        raise FileNotFoundError(f"Model BIN not found: {bin_path}")


def centroid_in_roi(centroid: tuple[int, int], roi: np.ndarray) -> bool:
    return cv2.pointPolygonTest(roi, centroid, False) >= 0


def compute_crop_rect(frame_shape: tuple[int, int], roi: np.ndarray, pad: int) -> tuple[int, int, int, int]:
    x, y, w, h = cv2.boundingRect(roi)
    x1 = max(0, x - pad)
    y1 = max(0, y - pad)
    x2 = min(frame_shape[1], x + w + pad)
    y2 = min(frame_shape[0], y + h + pad)
    return x1, y1, x2, y2


def time_weighted_average(samples: list[tuple[float, float]], window_start: float, now: float) -> float:
    if not samples:
        return 0.0
    if len(samples) == 1:
        return samples[0][1]

    total = 0.0
    weight = 0.0

    t0, v0 = samples[0]
    if t0 > window_start:
        dt = t0 - window_start
        total += v0 * dt
        weight += dt

    for i in range(1, len(samples)):
        t_prev, v_prev = samples[i - 1]
        t_curr, v_curr = samples[i]
        seg_start = max(t_prev, window_start)
        seg_end = min(t_curr, now)
        if seg_end <= seg_start:
            continue

        if t_curr != t_prev:
            v_start = v_prev + (v_curr - v_prev) * ((seg_start - t_prev) / (t_curr - t_prev))
            v_end = v_prev + (v_curr - v_prev) * ((seg_end - t_prev) / (t_curr - t_prev))
        else:
            v_start = v_prev
            v_end = v_curr

        dt = seg_end - seg_start
        total += (v_start + v_end) * 0.5 * dt
        weight += dt

    if weight <= 0.0:
        return samples[-1][1]
    return total / weight


def letterbox(img: np.ndarray, new_shape: int) -> tuple[np.ndarray, float, tuple[int, int]]:
    h, w = img.shape[:2]
    r = min(new_shape / h, new_shape / w)
    new_unpad = (int(round(w * r)), int(round(h * r)))
    dw = new_shape - new_unpad[0]
    dh = new_shape - new_unpad[1]
    left = dw // 2
    right = dw - left
    top = dh // 2
    bottom = dh - top

    resized = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
    padded = cv2.copyMakeBorder(
        resized, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(114, 114, 114)
    )
    return padded, r, (left, top)


def preprocess(frame: np.ndarray) -> tuple[np.ndarray, float, tuple[int, int]]:
    img, ratio, dwdh = letterbox(frame, MODEL_SIZE)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.astype(np.float32) / 255.0
    img = np.transpose(img, (2, 0, 1))[None, ...]
    return img, ratio, dwdh


def postprocess(
    preds: np.ndarray,
    ratio: float,
    dwdh: tuple[int, int],
    frame_shape: tuple[int, int],
) -> list[tuple[int, int, int, int, float]]:
    if preds.ndim == 3:
        preds = preds[0]
    if preds.shape[0] < preds.shape[1]:
        preds = preds.T

    boxes = preds[:, :4]
    scores = preds[:, 4:]
    class_ids = np.argmax(scores, axis=1)
    conf = scores[np.arange(scores.shape[0]), class_ids]

    keep = (conf >= CONF_THRESHOLD) & (class_ids == PERSON_CLASS_ID)
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

    boxes_xywh = []
    for (x1, y1, x2, y2) in boxes_xyxy:
        boxes_xywh.append([int(x1), int(y1), int(x2 - x1), int(y2 - y1)])

    idxs = cv2.dnn.NMSBoxes(boxes_xywh, conf.tolist(), CONF_THRESHOLD, NMS_THRESHOLD)
    if len(idxs) == 0:
        return []

    results = []
    for i in idxs.flatten():
        x, y, bw, bh = boxes_xywh[i]
        results.append((x, y, bw, bh, float(conf[i])))
    return results


def main() -> None:
    cap = cv2.VideoCapture(HLS_URL)
    if not cap.isOpened():
        raise RuntimeError("Failed to open HLS stream. Check network/FFmpeg support.")

    try:
        ensure_model_files(MODEL_PATH)
    except FileNotFoundError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        print(
            "Hint: export YOLOv8 to OpenVINO and set MODEL_PATH to the .xml file.",
            file=sys.stderr,
        )
        raise

    try:
        core = ov.Core()
        model = core.read_model(MODEL_PATH)
        compiled = core.compile_model(model, DEVICE)
        input_layer = compiled.input(0)
        output_layer = compiled.output(0)
    except Exception as exc:
        print(f"ERROR: OpenVINO init/compile failed: {exc}", file=sys.stderr)
        raise

    ok, frame = cap.read()
    if not ok:
        raise RuntimeError("Failed to read first frame from stream.")

    if SCALE != 1.0:
        frame = cv2.resize(frame, None, fx=SCALE, fy=SCALE, interpolation=cv2.INTER_AREA)
        roi = scale_points(ROI, SCALE)
    else:
        roi = ROI
    crop_rect = compute_crop_rect(frame.shape, roi, CROP_PAD)

    frame_count = 0
    last_print = 0.0
    last_infer = 0.0
    count_samples: list[tuple[float, float]] = []
    latest_smooth = 0.0

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                time.sleep(0.1)
                continue

            frame_count += 1
            if FRAME_STRIDE > 1 and (frame_count % FRAME_STRIDE != 0):
                continue

            if SCALE != 1.0:
                frame = cv2.resize(frame, None, fx=SCALE, fy=SCALE, interpolation=cv2.INTER_AREA)

            now = time.time()
            if INFER_EVERY_SEC > 0 and (now - last_infer) < INFER_EVERY_SEC:
                if SHOW_PREVIEW:
                    overlay = frame.copy()
                    cv2.polylines(overlay, [roi], True, (0, 255, 255), 1)
                    if CROP_TO_ROI:
                        x1, y1, x2, y2 = crop_rect
                        cv2.rectangle(overlay, (x1, y1), (x2, y2), (255, 255, 0), 1)
                    cv2.imshow("people", overlay)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        break
                continue
            last_infer = now

            if CROP_TO_ROI:
                x1, y1, x2, y2 = crop_rect
                if x2 <= x1 or y2 <= y1:
                    continue
                crop = frame[y1:y2, x1:x2]
                input_tensor, ratio, dwdh = preprocess(crop)
                infer_shape = crop.shape
            else:
                x1, y1 = 0, 0
                input_tensor, ratio, dwdh = preprocess(frame)
                infer_shape = frame.shape
            result = compiled({input_layer: input_tensor})
            preds = result[output_layer]
            detections = postprocess(preds, ratio, dwdh, infer_shape)

            in_roi = []
            for (x, y, w, h, conf) in detections:
                x += x1
                y += y1
                cx = int(x + w / 2)
                cy = int(y + h / 2)
                if centroid_in_roi((cx, cy), roi):
                    in_roi.append((x, y, w, h, conf))

            count_samples.append((now, float(len(in_roi))))
            window_start = now - SMOOTH_SECONDS
            while len(count_samples) > 2 and count_samples[1][0] <= window_start:
                count_samples.pop(0)
            latest_smooth = time_weighted_average(count_samples, window_start, now)

            if now - last_print >= 0.5:
                last_print = now
                stamp = time.strftime("%H:%M:%S")
                print(f"[{stamp}] people in ROI: {len(in_roi)} | smooth: {latest_smooth:.2f}")

            if SHOW_PREVIEW:
                overlay = frame.copy()
                cv2.polylines(overlay, [roi], True, (0, 255, 255), 1)
                if CROP_TO_ROI:
                    x1, y1, x2, y2 = crop_rect
                    cv2.rectangle(overlay, (x1, y1), (x2, y2), (255, 255, 0), 1)
                for (x, y, w, h, conf) in in_roi:
                    cv2.rectangle(overlay, (x, y), (x + w, y + h), (0, 255, 0), 1)
                    cv2.putText(
                        overlay,
                        f"{conf:.2f}",
                        (x, max(20, y - 6)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (0, 255, 0),
                        1,
                        cv2.LINE_AA,
                    )
                cv2.putText(
                    overlay,
                    f"count {len(in_roi)} | smooth {latest_smooth:.2f}",
                    (10, 100),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 255),
                    2,
                    cv2.LINE_AA,
                )
                cv2.imshow("people", overlay)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
