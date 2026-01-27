"""Postprocessing utilities for inference outputs."""

from __future__ import annotations

import numpy as np

from tracknet.core.config.constants import HEIGHT, WIDTH


def heatmap_to_xy(heat: np.ndarray) -> tuple[int, int, int]:
    """Decode a single heatmap to (x, y, v) using argmax.

    Visibility heuristic: visible if (x > 0 or y > 0).
    """

    heat = np.asarray(heat)
    if heat.shape != (HEIGHT, WIDTH):
        raise ValueError(f"expected heatmap shape ({HEIGHT}, {WIDTH}), got {heat.shape}")

    idx = int(np.argmax(heat.reshape(-1)))
    y0 = idx // int(WIDTH)
    x0 = idx % int(WIDTH)
    v = 1 if (x0 > 0 or y0 > 0) else 0
    return int(x0), int(y0), int(v)
