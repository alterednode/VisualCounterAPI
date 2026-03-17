from __future__ import annotations

from typing import Iterable

import cv2
import numpy as np

from visualcounter.config import Point
from visualcounter.models import Detection


def parse_roi_string(raw: str) -> list[Point]:
    points: list[Point] = []
    for token in raw.split(";"):
        token = token.strip()
        if not token:
            continue
        parts = token.split(",")
        if len(parts) != 2:
            raise ValueError(f"Invalid ROI point '{token}', expected x,y")
        x, y = parts
        x_f = float(x)
        y_f = float(y)
        if not (0.0 <= x_f <= 1.0 and 0.0 <= y_f <= 1.0):
            raise ValueError(f"ROI point '{token}' must be normalized between 0 and 1")
        points.append((x_f, y_f))

    if len(points) < 3:
        raise ValueError("ROI must contain at least 3 points")
    return points


def roi_polygon(roi_points: list[Point], frame_shape: tuple[int, int]) -> np.ndarray:
    frame_h, frame_w = frame_shape
    if frame_h <= 0 or frame_w <= 0:
        raise ValueError(f"Invalid frame shape {frame_shape}")

    max_x = max(frame_w - 1, 0)
    max_y = max(frame_h - 1, 0)
    pixel_points: list[tuple[int, int]] = []
    for x_norm, y_norm in roi_points:
        pixel_points.append((int(round(x_norm * max_x)), int(round(y_norm * max_y))))

    return np.array(pixel_points, dtype=np.int32)


def count_in_polygon(detections: Iterable[Detection], polygon: np.ndarray) -> int:
    count = 0
    for det in detections:
        cx, cy = det.centroid
        if cv2.pointPolygonTest(polygon, (cx, cy), False) >= 0:
            count += 1
    return count


def count_in_roi(
    detections: Iterable[Detection],
    roi_points: list[Point],
    frame_shape: tuple[int, int],
) -> int:
    polygon = roi_polygon(roi_points, frame_shape)
    return count_in_polygon(detections, polygon)


def roi_to_key(roi_name: str | None, roi_points: list[Point]) -> str:
    if roi_name:
        return f"name:{roi_name}"
    joined = ";".join(f"{x:.6g},{y:.6g}" for x, y in roi_points)
    return f"points:{joined}"


def _clip_polygon_against_edge(
    polygon: list[Point],
    inside: callable,
    intersect: callable,
) -> list[Point]:
    if not polygon:
        return []

    output: list[Point] = []
    prev = polygon[-1]
    prev_inside = inside(prev)
    for curr in polygon:
        curr_inside = inside(curr)
        if curr_inside:
            if not prev_inside:
                output.append(intersect(prev, curr))
            output.append(curr)
        elif prev_inside:
            output.append(intersect(prev, curr))
        prev = curr
        prev_inside = curr_inside
    return output


def clip_roi_to_unit_square(roi_points: list[Point]) -> list[Point]:
    # Clip polygon against x>=0, x<=1, y>=0, y<=1.
    poly: list[Point] = list(roi_points)

    poly = _clip_polygon_against_edge(
        poly,
        inside=lambda p: p[0] >= 0.0,
        intersect=lambda a, b: (0.0, a[1] + (b[1] - a[1]) * ((0.0 - a[0]) / (b[0] - a[0]))),
    )
    poly = _clip_polygon_against_edge(
        poly,
        inside=lambda p: p[0] <= 1.0,
        intersect=lambda a, b: (1.0, a[1] + (b[1] - a[1]) * ((1.0 - a[0]) / (b[0] - a[0]))),
    )
    poly = _clip_polygon_against_edge(
        poly,
        inside=lambda p: p[1] >= 0.0,
        intersect=lambda a, b: (a[0] + (b[0] - a[0]) * ((0.0 - a[1]) / (b[1] - a[1])), 0.0),
    )
    poly = _clip_polygon_against_edge(
        poly,
        inside=lambda p: p[1] <= 1.0,
        intersect=lambda a, b: (a[0] + (b[0] - a[0]) * ((1.0 - a[1]) / (b[1] - a[1])), 1.0),
    )

    deduped: list[Point] = []
    for point in poly:
        if deduped and abs(point[0] - deduped[-1][0]) < 1e-9 and abs(point[1] - deduped[-1][1]) < 1e-9:
            continue
        deduped.append(point)

    if len(deduped) >= 2:
        first = deduped[0]
        last = deduped[-1]
        if abs(first[0] - last[0]) < 1e-9 and abs(first[1] - last[1]) < 1e-9:
            deduped.pop()

    return deduped


def transform_roi_for_source_crop(
    roi_points: list[Point],
    source_crop: tuple[float, float, float, float] | None,
) -> list[Point]:
    if source_crop is None:
        return list(roi_points)

    x1, y1, x2, y2 = source_crop
    width = x2 - x1
    height = y2 - y1
    if width <= 0.0 or height <= 0.0:
        return []

    transformed = [((x - x1) / width, (y - y1) / height) for x, y in roi_points]
    clipped = clip_roi_to_unit_square(transformed)
    if len(clipped) < 3:
        return []
    return clipped
