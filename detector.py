from dataclasses import dataclass

import cv2
import numpy as np
_TYPICAL_ITEM_ASPECT = 0.3


@dataclass
class BBox:
    x: int
    y: int
    w: int
    h: int
    count: int = 1

    @property
    def center(self):
        return (self.x + self.w / 2, self.y + self.h / 2)

    @property
    def area(self):
        return self.w * self.h


def _remove_long_horizontal_lines(edges: np.ndarray, min_length: int) -> np.ndarray:
    horiz_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (min_length, 1))
    horiz_lines = cv2.morphologyEx(edges, cv2.MORPH_OPEN, horiz_kernel, iterations=1)
    horiz_lines = cv2.dilate(horiz_lines, np.ones((3, 3), np.uint8), iterations=1)
    return cv2.bitwise_and(edges, cv2.bitwise_not(horiz_lines))


def _find_occupied_ranges(
    profile: np.ndarray,
    thresh_frac: float,
    min_len_frac: float,
    gap_merge: int,
) -> list[tuple[int, int]]:
    total_len = len(profile)
    smooth = np.convolve(profile, np.ones(3) / 3, mode="same")
    thresh = max(5.0, thresh_frac * smooth.max())
    occupied = smooth > thresh

    raw_ranges: list[list[int]] = []
    start = None
    for i, is_occupied in enumerate(occupied):
        if is_occupied and start is None:
            start = i
        elif not is_occupied and start is not None:
            raw_ranges.append([start, i])
            start = None
    if start is not None:
        raw_ranges.append([start, total_len])
    merged: list[list[int]] = []
    for r in raw_ranges:
        if merged and r[0] - merged[-1][1] <= gap_merge:
            merged[-1][1] = r[1]
        else:
            merged.append(r)

    min_len = min_len_frac * total_len
    return [(r[0], r[1]) for r in merged if (r[1] - r[0]) >= min_len]


def _split_by_color(
    image: np.ndarray,
    x1: int,
    x2: int,
    diff_threshold: float = 0.22,
    window: int = 45,
    min_segment_frac: float = 0.08,
) -> list[tuple[int, int]]:
    segment = image[:, x1:x2]
    h = segment.shape[0]
    if h < 10 or (x2 - x1) < window * 2:
        return [(x1, x2)]

    hsv = cv2.cvtColor(segment, cv2.COLOR_BGR2HSV)
    strip = hsv[int(h * 0.35) : int(h * 0.65), :, :].astype(np.float32)
    hue, sat, val = strip[:, :, 0], strip[:, :, 1], strip[:, :, 2]

    weighted_hue = (hue * sat).sum(axis=0) / np.maximum(sat.sum(axis=0), 1.0)
    mean_sat = sat.mean(axis=0)
    mean_val = val.mean(axis=0)

    w = len(weighted_hue)
    if w < window * 2:
        return [(x1, x2)]

    def smooth(arr, k=15):
        return np.convolve(arr, np.ones(k) / k, mode="same")

    feat_hue = smooth(weighted_hue) / 180.0
    feat_sat = smooth(mean_sat) / 255.0
    feat_val = smooth(mean_val) / 255.0
    features = np.stack([feat_hue, feat_sat, feat_val], axis=1)

    diff = np.zeros(w)
    for x in range(window, w - window):
        left = np.median(features[x - window : x], axis=0)
        right = np.median(features[x : x + window], axis=0)
        diff[x] = float(np.linalg.norm(right - left))

    candidates = np.where(diff > diff_threshold)[0]
    splits: list[int] = []
    min_gap = window
    for c in candidates:
        if not splits or c - splits[-1] > min_gap:
            splits.append(int(c))
        elif diff[c] > diff[splits[-1]]:
            splits[-1] = int(c)

    if not splits:
        return [(x1, x2)]

    boundaries = [0] + splits + [w]
    min_len = min_segment_frac * w
    segments = []
    for i in range(len(boundaries) - 1):
        seg_start, seg_end = boundaries[i], boundaries[i + 1]
        if seg_end - seg_start < min_len:
            if segments:
                segments[-1] = (segments[-1][0], seg_end)
            else:
                boundaries[i + 1] = boundaries[i]
            continue
        segments.append((seg_start, seg_end))

    if not segments:
        return [(x1, x2)]

    return [(x1 + s, x1 + e) for s, e in segments]


def _estimate_count(width: int, height: int) -> int:
    if height <= 0:
        return 1
    single_item_width = height * _TYPICAL_ITEM_ASPECT
    if single_item_width <= 0:
        return 1
    return max(1, round(width / single_item_width))


def detect_products(
    image: np.ndarray,
    min_area_ratio: float = 0.002,
    max_area_ratio: float = 0.8,
) -> list[BBox]:
    h_img, w_img = image.shape[:2]
    img_area = h_img * w_img

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(gray, 30, 100)

    horiz_min_length = max(60, int(w_img * 0.6))
    edges_clean = _remove_long_horizontal_lines(edges, horiz_min_length)

    col_profile = (edges_clean > 0).sum(axis=0).astype(np.float32)
    col_ranges = _find_occupied_ranges(
        col_profile, thresh_frac=0.12, min_len_frac=0.03, gap_merge=5
    )
    color_split_ranges: list[tuple[int, int]] = []
    for x1, x2 in col_ranges:
        color_split_ranges.extend(_split_by_color(image, x1, x2))

    boxes: list[BBox] = []
    for x1, x2 in color_split_ranges:
        sub = edges_clean[:, x1:x2]
        row_profile = (sub > 0).sum(axis=1).astype(np.float32)
        row_ranges = _find_occupied_ranges(
            row_profile, thresh_frac=0.15, min_len_frac=0.05, gap_merge=60
        )
        for y1, y2 in row_ranges:
            w, h = x2 - x1, y2 - y1
            ratio = (w * h) / img_area
            if ratio < min_area_ratio or ratio > max_area_ratio:
                continue
            if h < 0.08 * h_img:
                continue
            aspect = w / h if h else 0
            if aspect < 0.1 or aspect > 6.0:
                continue
            count = _estimate_count(w, h)
            boxes.append(BBox(x1, y1, w, h, count=count))

    return boxes


def auto_crop(image: np.ndarray, sharpness_thresh_frac: float = 0.15) -> tuple[int, int, int, int]:
    h_img, w_img = image.shape[:2]
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    lap = cv2.Laplacian(gray, cv2.CV_64F)
    band = lap[h_img // 3 : 2 * h_img // 3, :]
    col_sharpness = band.var(axis=0)
    col_smooth = np.convolve(col_sharpness, np.ones(21) / 21, mode="same")

    col_thresh = sharpness_thresh_frac * col_smooth.max()
    x_focus = np.where(col_smooth > col_thresh)[0]

    if len(x_focus) < 0.2 * w_img:
       
        x1, x2 = 0, w_img
    else:
        x1, x2 = int(x_focus.min()), int(x_focus.max())

    row_band = lap[:, x1:x2] if x2 > x1 else lap
    row_sharpness = row_band.var(axis=1)
    row_smooth = np.convolve(row_sharpness, np.ones(15) / 15, mode="same")
    row_thresh = sharpness_thresh_frac * row_smooth.max()
    y_focus = np.where(row_smooth > row_thresh)[0]

    if len(y_focus) < 0.2 * h_img:
        y1, y2 = 0, h_img
    else:
        y1, y2 = int(y_focus.min()), int(y_focus.max())

    return x1, y1, x2, y2
