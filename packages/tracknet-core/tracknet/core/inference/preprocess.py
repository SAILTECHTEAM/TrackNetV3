"""Preprocessing helpers shared by TrackNet inference wrappers."""

from __future__ import annotations

"""Preprocessing helpers shared by TrackNet inference wrappers."""

import time
from collections import deque
from typing import Deque

import cv2
import numpy as np

from tracknet.core.config.constants import HEIGHT, WIDTH


class Preprocessor:
    """Encapsulates the TrackNet preprocess stack (resizing, median, difference, CHW casts)."""

    def __init__(
        self,
        seq_len: int,
        bg_mode: str,
        *,
        median: np.ndarray | None = None,
        median_warmup: int = 0,
        stats: dict[str, float] | None = None,
    ):
        self.seq_len = int(seq_len)
        self.bg_mode = (bg_mode or "").strip()

        self.stats = stats or {}
        self.stats.setdefault("t_median", 0.0)

        self._median = median
        self._median_warmup = int(median_warmup)
        self._warmup_rgb: list[np.ndarray] = []

        self.proc: Deque[np.ndarray] = deque(maxlen=self.seq_len)

        self.img_scaler: tuple[float, float] | None = None
        self.img_shape: tuple[int, int] | None = None

    @property
    def median(self) -> np.ndarray | None:
        return self._median

    def reset(self) -> None:
        self._warmup_rgb.clear()
        self.proc.clear()
        self.img_scaler = None
        self.img_shape = None

    def ensure_scaler(self, frame_bgr: np.ndarray) -> None:
        if self.img_scaler is not None:
            return
        h, w = frame_bgr.shape[:2]
        self.img_shape = (int(w), int(h))
        self.img_scaler = (float(w) / float(WIDTH), float(h) / float(HEIGHT))

    def _maybe_build_median(self, rgb_resized: np.ndarray) -> None:
        if not self.bg_mode or self._median is not None:
            return
        if self._median_warmup <= 0:
            raise RuntimeError(
                f"bg_mode='{self.bg_mode}' needs median. Pass median=... or set median_warmup>0."
            )

        self._warmup_rgb.append(rgb_resized)
        if len(self._warmup_rgb) >= self._median_warmup:
            t0 = time.perf_counter()
            med = np.median(np.stack(self._warmup_rgb, axis=0), axis=0).astype(np.uint8)
            self.stats["t_median"] += time.perf_counter() - t0

            if self.bg_mode == "concat":
                self._median = np.moveaxis(med, -1, 0)
            else:
                self._median = med
            self._warmup_rgb.clear()

    def process_one(self, frame_bgr: np.ndarray) -> np.ndarray | None:
        """Process a single BGR frame into a feature tensor.

        Returns None while waiting for median warmup if the selected bg_mode requires it.
        """

        rgb = frame_bgr[..., ::-1]
        rgb_resized = cv2.resize(rgb, (WIDTH, HEIGHT), interpolation=cv2.INTER_AREA)

        if self.bg_mode and self._median is None:
            self._maybe_build_median(rgb_resized)
            if self._median is None:
                return None

        if self.bg_mode == "subtract":
            median = self._median
            assert median is not None
            diff = np.abs(rgb_resized.astype(np.int16) - median.astype(np.int16)).sum(axis=2)
            diff = np.clip(diff, 0, 255).astype(np.uint8)
            diff_f = diff.astype(np.float32) / 255.0
            return diff_f[None, ...]

        if self.bg_mode == "subtract_concat":
            median = self._median
            assert median is not None
            diff = np.abs(rgb_resized.astype(np.int16) - median.astype(np.int16)).sum(axis=2)
            diff = np.clip(diff, 0, 255).astype(np.uint8)
            img_f = rgb_resized.astype(np.float32) / 255.0
            diff_f = diff.astype(np.float32) / 255.0
            img_chw = np.moveaxis(img_f, -1, 0)
            return np.concatenate([img_chw, diff_f[None]], axis=0)

        img_f = rgb_resized.astype(np.float32) / 255.0
        return np.moveaxis(img_f, -1, 0)
