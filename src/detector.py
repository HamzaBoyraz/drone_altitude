import cv2
import numpy as np
import math
from typing import Dict, Optional


def filter_by_color(
    image: np.ndarray, 
    lower_hsv: np.ndarray, 
    upper_hsv: np.ndarray,
    use_morphology: bool = True,
    kernel_size: int = 5
) -> np.ndarray:
    """
    Converts BGR to HSV and applies thresholding.
    Uses an ELLIPTICAL kernel instead of a square kernel to preserve organic curves.
    Set use_morphology=False to disable post-processing entirely.
    """
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv_image, lower_hsv, upper_hsv)

    if use_morphology and kernel_size > 0:
        # Use MORPH_ELLIPSE for circular structuring element instead of square np.ones()
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    return mask


def extract_primary_contour(
    binary_mask: np.ndarray, 
    min_area: float = 300.0
) -> Optional[np.ndarray]:
    """
    Detects contours using CHAIN_APPROX_NONE to preserve every exact perimeter point
    without corner approximation.
    """
    # CHAIN_APPROX_NONE stores all contour points (no straight line approximation)
    contours, _ = cv2.findContours(
        binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
    )

    best_contour = None
    best_score = -1.0

    for contour in contours:
        area = cv2.contourArea(contour)
        if area < min_area:
            continue

        perimeter = cv2.arcLength(contour, closed=True)
        if perimeter == 0:
            continue

        circularity = (4.0 * math.pi * area) / (perimeter ** 2)
        score = circularity * math.log(area)

        if score > best_score:
            best_score = score
            best_contour = contour

    return best_contour


def get_contour_metrics(contour: np.ndarray) -> Optional[Dict[str, any]]:
    area = cv2.contourArea(contour)
    moments = cv2.moments(contour)

    if moments["m00"] == 0:
        return None

    cx = int(moments["m10"] / moments["m00"])
    cy = int(moments["m01"] / moments["m00"])

    return {
        "center": (cx, cy),
        "area_px": area
    }