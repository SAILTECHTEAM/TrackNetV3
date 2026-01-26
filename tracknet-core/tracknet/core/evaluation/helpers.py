"""Evaluation helpers.

This module is intentionally lightweight and shared across runtimes.
"""

from __future__ import annotations

import numpy as np


def binarize_heatmap(heatmap: np.ndarray, threshold: float = 0.5) -> np.ndarray:
    """Convert a float heatmap into a uint8 mask suitable for contour finding."""

    heatmap = np.asarray(heatmap)
    if heatmap.size == 0:
        return heatmap.astype(np.uint8)
    m = heatmap.max()
    if m <= 0:
        return np.zeros_like(heatmap, dtype=np.uint8)
    norm = heatmap / m
    return (norm >= threshold).astype(np.uint8)
