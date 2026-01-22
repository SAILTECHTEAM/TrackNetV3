"""Prediction utilities for TrackNetV3."""

import cv2
import numpy as np


def predict_location(heatmap):
    """Get coordinates from the heatmap.

    Args:
        heatmap (numpy.ndarray): A single heatmap with shape (H, W)

    Returns:
        x, y, w, h (Tuple[int, int, int, int]): bounding box of the the bounding box with max area
    """
    if np.amax(heatmap) == 0:
        return 0, 0, 0, 0
    else:
        (cnts, _) = cv2.findContours(heatmap.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        rects = [cv2.boundingRect(ctr) for ctr in cnts]

        max_area_idx = 0
        max_area = rects[0][2] * rects[0][3]
        for i in range(1, len(rects)):
            area = rects[i][2] * rects[i][3]
            if area > max_area:
                max_area_idx = i
                max_area = area
        x, y, w, h = rects[max_area_idx]

        return x, y, w, h
