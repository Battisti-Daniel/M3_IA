from __future__ import annotations

import cv2
import numpy as np

from .data_structures import BBox


def crop_from_bbox(frame: np.ndarray, bbox: BBox) -> np.ndarray:
    x1, y1, x2, y2 = [int(v) for v in bbox]
    h, w = frame.shape[:2]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w - 1, x2), min(h - 1, y2)
    return frame[y1:y2, x1:x2].copy()


def estimate_plate_quadrilateral(mask: np.ndarray) -> np.ndarray:
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        h, w = mask.shape[:2]
        return np.array([[0, 0], [w, 0], [w, h], [0, h]], dtype=np.float32)
    contour = max(contours, key=cv2.contourArea)
    epsilon = 0.02 * cv2.arcLength(contour, True)
    approx = cv2.approxPolyDP(contour, epsilon, True)
    if len(approx) == 4:
        return approx.reshape(-1, 2).astype(np.float32)
    rect = cv2.boundingRect(contour)
    x, y, w, h = rect
    return np.array([[x, y], [x + w, y], [x + w, y + h], [x, y + h]], dtype=np.float32)


def correct_perspective(crop: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    _, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    quad = estimate_plate_quadrilateral(thresh)
    h, w = gray.shape[:2]
    dst = np.array([[0, 0], [w, 0], [w, h], [0, h]], dtype=np.float32)
    matrix = cv2.getPerspectiveTransform(quad, dst)
    warped = cv2.warpPerspective(crop, matrix, (w, h))
    return warped


def enhance_for_ocr(image: np.ndarray) -> np.ndarray:
    resized = cv2.resize(image, (320, 80), interpolation=cv2.INTER_CUBIC)
    lab = cv2.cvtColor(resized, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    l_enhanced = clahe.apply(l)
    lab_enhanced = cv2.merge((l_enhanced, a, b))
    enhanced = cv2.cvtColor(lab_enhanced, cv2.COLOR_LAB2BGR)
    kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
    sharpened = cv2.filter2D(enhanced, -1, kernel)
    denoised = cv2.bilateralFilter(sharpened, d=5, sigmaColor=50, sigmaSpace=50)
    return denoised
