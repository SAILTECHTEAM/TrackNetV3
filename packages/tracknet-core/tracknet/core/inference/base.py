"""Framework-agnostic inference base modules.

This module is numpy/cv2-only and intended to be shared by PyTorch/ONNX wrappers.
"""

from __future__ import annotations

import abc
import time
from collections import deque
from typing import Any

import numpy as np

from tracknet.core.config.constants import HEIGHT, WIDTH

from .ensemble import get_ensemble_weight
from .postprocessing import heatmap_to_xy
from .preprocess import Preprocessor


class BaseTrackNetModule(abc.ABC):
    """Streaming TrackNet inference (pre/post + temporal accumulation).

    Core constraints:
    - Numpy/cv2 only.
    - Preprocess produces float32 CHW in [0, 1] at (HEIGHT, WIDTH).
    - Ensemble accumulation uses weight indexing w[seq_len - 1 - j].
    """

    def __init__(
        self,
        seq_len: int,
        bg_mode: str,
        eval_mode: str = "weight",
        *,
        median: np.ndarray | None = None,
        median_warmup: int = 0,
    ):
        self.seq_len = int(seq_len)
        self.bg_mode = (bg_mode or "").strip()
        self.eval_mode = (eval_mode or "").strip()

        self._ens_w = np.asarray(
            get_ensemble_weight(self.seq_len, self.eval_mode), dtype=np.float32
        )
        self._acc_sum: dict[int, np.ndarray] = {}
        self._acc_w: dict[int, float] = {}

        self._pending: deque[dict[str, Any]] = deque()

        self._count = 0

        # Debug hooks for parity verification (populated when forward runs).
        self._last_x_np: np.ndarray | None = None
        self._last_y_np: np.ndarray | None = None

        self.stats = {
            "push_calls": 0,
            "outputs": 0,
            "t_preprocess": 0.0,
            "t_forward": 0.0,
            "t_post": 0.0,
            "t_median": 0.0,
        }

        self.preprocessor = Preprocessor(
            seq_len=self.seq_len,
            bg_mode=self.bg_mode,
            median=median,
            median_warmup=median_warmup,
            stats=self.stats,
        )
        self._proc: deque[np.ndarray] = self.preprocessor.proc
        self._fidq: deque[int] = deque(maxlen=self.seq_len)

    def reset(self) -> None:
        self._acc_sum.clear()
        self._acc_w.clear()
        self.preprocessor.reset()
        self._proc.clear()
        self._fidq.clear()
        self._count = 0
        self._last_x_np = None
        self._last_y_np = None
        self._pending.clear()

    @property
    def img_scaler(self) -> tuple[float, float] | None:
        return self.preprocessor.img_scaler

    @property
    def img_shape(self) -> tuple[int, int] | None:
        return self.preprocessor.img_shape

    def _decode_heatmap(self, heat: np.ndarray) -> tuple[int, int, int]:
        x0, y0, v0 = heatmap_to_xy(heat)
        if self.img_scaler is None:
            return x0, y0, v0
        sx, sy = self.img_scaler
        x = int(float(x0) * float(sx))
        y = int(float(y0) * float(sy))
        v = 1 if (x > 0 or y > 0) else 0
        return x, y, v

    @abc.abstractmethod
    def _forward(self, x_np: np.ndarray) -> np.ndarray:
        """Run model inference.

        Args:
            x_np: float32 array of shape (1, C, HEIGHT, WIDTH).

        Returns:
            y_np: float32 array of shape (seq_len, HEIGHT, WIDTH).
        """

    def push(self, frame_bgr: np.ndarray, frame_id: int | None = None) -> dict[str, Any] | None:
        """Push one frame; return one prediction when ready."""

        self.stats["push_calls"] += 1
        self.preprocessor.ensure_scaler(frame_bgr)
        if frame_id is None:
            frame_id = self._count
        self._count += 1

        t0 = time.perf_counter()
        feat = self.preprocessor.process_one(frame_bgr)
        self.stats["t_preprocess"] += time.perf_counter() - t0
        if feat is None:
            return None

        self._fidq.append(int(frame_id))

        self._proc.append(feat)
        if len(self._proc) < self.seq_len:
            return self._pending.popleft() if self._pending else None

        if self.bg_mode == "concat":
            median = self.preprocessor.median
            if median is None:
                raise RuntimeError("Preprocessor median missing for concat bg_mode")
            x_np = np.concatenate([median.astype(np.float32) / 255.0] + list(self._proc), axis=0)
        else:
            x_np = np.concatenate(list(self._proc), axis=0)

        self._last_x_np = x_np[None, ...].astype(np.float32)

        t1 = time.perf_counter()
        y = self._forward(self._last_x_np)
        self.stats["t_forward"] += time.perf_counter() - t1

        y = np.asarray(y, dtype=np.float32)
        if y.shape != (self.seq_len, HEIGHT, WIDTH):
            raise ValueError(
                f"_forward must return shape ({self.seq_len}, {HEIGHT}, {WIDTH}), got {y.shape}"
            )

        self._last_y_np = y

        t2 = time.perf_counter()
        ids = list(self._fidq)

        for j, fid in enumerate(ids):
            w = 1.0 if self.eval_mode == "nonoverlap" else float(self._ens_w[self.seq_len - 1 - j])
            if fid not in self._acc_sum:
                self._acc_sum[fid] = (y[j] * w).astype(np.float32)
                self._acc_w[fid] = w
            else:
                self._acc_sum[fid] += (y[j] * w).astype(np.float32)
                self._acc_w[fid] += w

        if self.eval_mode == "nonoverlap":
            # Window is non-overlapping: emit all seq_len outputs (one per push via _pending).
            for fid in ids:
                heat = self._acc_sum[fid] / max(1e-6, float(self._acc_w[fid]))
                x_out, y_out, v_out = self._decode_heatmap(heat)
                self._pending.append(
                    {"Frame": int(fid), "X": int(x_out), "Y": int(y_out), "Visibility": int(v_out)}
                )
                self.stats["outputs"] += 1
            self._acc_sum.clear()
            self._acc_w.clear()
            self._proc.clear()
            self._fidq.clear()

        out_id = int(frame_id) - (self.seq_len - 1)
        out = None
        if out_id in self._acc_sum:
            heat = self._acc_sum[out_id] / max(1e-6, float(self._acc_w[out_id]))
            x_out, y_out, v_out = self._decode_heatmap(heat)
            del self._acc_sum[out_id]
            del self._acc_w[out_id]
            out = {"Frame": int(out_id), "X": int(x_out), "Y": int(y_out), "Visibility": int(v_out)}
            self.stats["outputs"] += 1

        self.stats["t_post"] += time.perf_counter() - t2
        return out if out is not None else (self._pending.popleft() if self._pending else None)

    def flush(self) -> list[dict[str, Any]]:
        outs = []
        for fid in sorted(self._acc_sum.keys()):
            heat = self._acc_sum[fid] / max(1e-6, float(self._acc_w[fid]))
            x_out, y_out, v_out = self._decode_heatmap(heat)
            outs.append(
                {"Frame": int(fid), "X": int(x_out), "Y": int(y_out), "Visibility": int(v_out)}
            )
        self._acc_sum.clear()
        self._acc_w.clear()
        self._proc.clear()
        self._fidq.clear()
        return outs


class BaseInpaintModule(abc.ABC):
    """Streaming inpaint core logic (numpy-only).

    The base class implements per-sample normalization/denormalization and blending.
    Subclasses are responsible for model-specific batching. For example, an ONNX
    subclass may internally expand inputs to batch_size=4 in `_forward`.
    """

    def __init__(self, seq_len: int, *, img_scaler: tuple[float, float] | None = None):
        self.seq_len = int(seq_len)
        self.img_scaler = img_scaler

        self._coords: deque[list[float]] = deque(maxlen=self.seq_len)
        self._mask: deque[list[float]] = deque(maxlen=self.seq_len)
        self._frame_ids: deque[int] = deque(maxlen=self.seq_len)

        self.stats = {"push_calls": 0, "outputs": 0, "t_forward": 0.0, "t_post": 0.0}

    def reset(self) -> None:
        self._coords.clear()
        self._mask.clear()
        self._frame_ids.clear()

    @abc.abstractmethod
    def _forward(self, coords: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """Run model inference.

        Args:
            coords: float32 array of shape (B, seq_len, 2)
            mask: float32 array of shape (B, seq_len, 1)

        Returns:
            float32 array of shape (B, seq_len, 2)
        """

    @staticmethod
    def _normalize_one(
        x: int, y: int, vis: int, img_scaler: tuple[float, float]
    ) -> tuple[float, float, float]:
        sx, sy = img_scaler
        if int(vis) == 1 and (int(x) > 0 or int(y) > 0):
            cx = float(x) / (float(WIDTH) * float(sx))
            cy = float(y) / (float(HEIGHT) * float(sy))
            cx = min(max(cx, 0.0), 1.0)
            cy = min(max(cy, 0.0), 1.0)
            m = 0.0
        else:
            cx, cy = 0.0, 0.0
            m = 1.0
        return cx, cy, m

    @staticmethod
    def _denormalize_one(cx: float, cy: float, img_scaler: tuple[float, float]) -> tuple[int, int]:
        sx, sy = img_scaler
        px = int(float(cx) * float(WIDTH) * float(sx))
        py = int(float(cy) * float(HEIGHT) * float(sy))
        return px, py

    def push(
        self, pred: dict[str, Any], *, img_scaler: tuple[float, float] | None = None
    ) -> dict | None:
        self.stats["push_calls"] += 1

        if img_scaler is not None:
            self.img_scaler = img_scaler
        if self.img_scaler is None:
            raise ValueError(
                "img_scaler must be provided (sx = frame_w/WIDTH, sy = frame_h/HEIGHT)"
            )

        fid = int(pred["Frame"])
        x = int(pred["X"])
        y = int(pred["Y"])
        vis = int(pred["Visibility"])

        cx, cy, m = self._normalize_one(x, y, vis, self.img_scaler)
        self._coords.append([cx, cy])
        self._mask.append([m])
        self._frame_ids.append(fid)

        if len(self._coords) < self.seq_len:
            return None

        coor = np.asarray(self._coords, dtype=np.float32)[None, ...]
        mask = np.asarray(self._mask, dtype=np.float32)[None, ...]

        t1 = time.perf_counter()
        out = self._forward(coor, mask)
        self.stats["t_forward"] += time.perf_counter() - t1

        t2 = time.perf_counter()
        out = np.asarray(out, dtype=np.float32)
        if out.shape != coor.shape:
            raise ValueError(f"_forward must return shape {coor.shape}, got {out.shape}")

        # Blend: out = out * mask + coor * (1.0 - mask)
        out0 = out * mask + coor * (1.0 - mask)
        cx_out, cy_out = float(out0[0, 0, 0]), float(out0[0, 0, 1])

        out_fid = int(self._frame_ids[0])
        px, py = self._denormalize_one(cx_out, cy_out, self.img_scaler)
        v = 1 if (px > 0 or py > 0) else 0

        self.stats["t_post"] += time.perf_counter() - t2
        self.stats["outputs"] += 1

        return {"Frame": out_fid, "X": int(px), "Y": int(py), "Visibility": int(v)}

    def flush(self) -> list[dict[str, Any]]:
        outs: list[dict[str, Any]] = []
        if self.img_scaler is None:
            return outs
        while self._frame_ids:
            fid = self._frame_ids.popleft()
            c = self._coords.popleft()
            m = self._mask.popleft()
            if m[0] >= 0.5:
                px, py, v = 0, 0, 0
            else:
                px, py = self._denormalize_one(c[0], c[1], self.img_scaler)
                v = 1 if (px > 0 or py > 0) else 0
            outs.append({"Frame": int(fid), "X": int(px), "Y": int(py), "Visibility": int(v)})
        return outs
