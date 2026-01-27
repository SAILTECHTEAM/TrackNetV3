from __future__ import annotations

import os
import time
from collections import deque
from typing import Any

import cv2
import numpy as np
import onnxruntime as ort
from tracknet.core.config.constants import HEIGHT, WIDTH

from .helpers import _ensemble_weights


class TrackNetModule:
    """Streaming TrackNet inference with ONNX Runtime."""

    def __init__(
        self,
        model_path: str,
        seq_len: int,
        bg_mode: str,
        device: Any = None,
        eval_mode: str = "weight",
        median: np.ndarray | None = None,
        median_warmup: int = 0,
    ):
        if not os.path.exists(model_path):
            raise ValueError(f"ONNX model file not found: {model_path}")

        sess_options = ort.SessionOptions()
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        sess_options.intra_op_num_threads = 1

        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        self.session = ort.InferenceSession(
            model_path, providers=providers, sess_options=sess_options
        )

        if not self.session.get_providers():
            raise RuntimeError("No ONNX Runtime execution providers available")

        self.input_name = self.session.get_inputs()[0].name
        self.output_names = [output.name for output in self.session.get_outputs()]

        # Validate metadata
        model_metadata = self.session.get_modelmeta().custom_metadata_map
        if "seq_len" in model_metadata:
            m_seq_len = int(model_metadata["seq_len"])
            if m_seq_len != int(seq_len):
                raise ValueError(
                    f"Model metadata mismatch: expected seq_len={seq_len}, got {m_seq_len}. Model={model_path}"
                )

        if bg_mode and "bg_mode" in model_metadata:
            m_bg_mode = model_metadata["bg_mode"]
            if m_bg_mode != bg_mode:
                raise ValueError(
                    f"Model metadata mismatch: expected bg_mode='{bg_mode}', got '{m_bg_mode}'. Model={model_path}"
                )

        self.seq_len = int(seq_len)
        self.bg_mode = bg_mode or ""
        self.device = device  # For API compatibility
        self.eval_mode = eval_mode

        self.img_scaler: tuple[float, float] | None = None
        self.img_shape: tuple[int, int] | None = None

        self._ens_w = _ensemble_weights(self.seq_len, self.eval_mode)
        self._acc_sum: dict[int, np.ndarray] = {}
        self._acc_w: dict[int, float] = {}

        self._median = median
        self._median_warmup = int(median_warmup)
        self._warmup_rgb: list[np.ndarray] = []
        self._proc = deque(maxlen=self.seq_len)
        self._fidq = deque(maxlen=self.seq_len)

        self.stats = {
            "push_calls": 0,
            "outputs": 0,
            "t_preprocess": 0.0,
            "t_forward": 0.0,
            "t_post": 0.0,
            "t_median": 0.0,
        }

        self._count = 0

    def reset(self):
        self._acc_sum.clear()
        self._acc_w.clear()
        self._warmup_rgb.clear()
        self._proc.clear()
        self._fidq.clear()
        self._count = 0
        self.img_scaler = None
        self.img_shape = None

    def _ensure_scaler(self, frame_bgr: np.ndarray):
        if self.img_scaler is None:
            h, w = frame_bgr.shape[:2]
            self.img_shape = (w, h)
            self.img_scaler = (w / WIDTH, h / HEIGHT)

    def _maybe_build_median(self, rgb_frame: np.ndarray):
        if not self.bg_mode:
            return
        if self._median is not None:
            return
        if self._median_warmup <= 0:
            raise RuntimeError(
                f"bg_mode='{self.bg_mode}' needs median. Pass median=... or set median_warmup>0."
            )

        self._warmup_rgb.append(rgb_frame)
        if len(self._warmup_rgb) >= self._median_warmup:
            t0 = time.perf_counter()
            med = np.median(np.stack(self._warmup_rgb, axis=0), axis=0).astype(np.uint8)
            self.stats["t_median"] += time.perf_counter() - t0
            if self.bg_mode == "concat":
                med_r = cv2.resize(med, (WIDTH, HEIGHT), interpolation=cv2.INTER_AREA)
                self._median = np.moveaxis(med_r, -1, 0)
            else:
                self._median = med
            self._warmup_rgb.clear()

    def _process_one(self, frame_bgr: np.ndarray) -> np.ndarray | None:
        rgb = frame_bgr[..., ::-1]

        if self.bg_mode and self._median is None:
            self._maybe_build_median(rgb)
            if self._median is None:
                return None

        if self.bg_mode == "subtract":
            diff = np.abs(rgb.astype(np.int16) - self._median.astype(np.int16)).sum(axis=2)
            diff = np.clip(diff, 0, 255).astype(np.uint8)
            diff_r = (
                cv2.resize(diff, (WIDTH, HEIGHT), interpolation=cv2.INTER_AREA).astype(np.float32)
                / 255.0
            )
            return diff_r[None, ...]

        if self.bg_mode == "subtract_concat":
            diff = np.abs(rgb.astype(np.int16) - self._median.astype(np.int16)).sum(axis=2)
            diff = np.clip(diff, 0, 255).astype(np.uint8)
            img_r = (
                cv2.resize(rgb, (WIDTH, HEIGHT), interpolation=cv2.INTER_AREA).astype(np.float32)
                / 255.0
            )
            diff_r = (
                cv2.resize(diff, (WIDTH, HEIGHT), interpolation=cv2.INTER_AREA).astype(np.float32)
                / 255.0
            )
            img_chw = np.moveaxis(img_r, -1, 0)
            return np.concatenate([img_chw, diff_r[None]], 0)

        img_r = (
            cv2.resize(rgb, (WIDTH, HEIGHT), interpolation=cv2.INTER_AREA).astype(np.float32)
            / 255.0
        )
        return np.moveaxis(img_r, -1, 0)

    def _heatmap_to_xy(self, heat: np.ndarray) -> tuple[int, int, int]:
        if heat is None or float(heat.max()) <= 0.0:
            return 0, 0, 0
        hm_h, hm_w = heat.shape
        idx = int(np.argmax(heat.reshape(-1)))
        y0 = idx // hm_w
        x0 = idx % hm_w
        sx, sy = self.img_scaler
        x = int(x0 * sx)
        y = int(y0 * sy)
        v = 0 if (x == 0 and y == 0) else 1
        return x, y, v

    def push(self, frame_bgr: np.ndarray, frame_id: int | None = None) -> dict[str, Any] | None:
        self.stats["push_calls"] += 1

        self._ensure_scaler(frame_bgr)
        if frame_id is None:
            frame_id = self._count
        self._count += 1

        self._fidq.append(frame_id)

        t0 = time.perf_counter()
        feat = self._process_one(frame_bgr)
        self.stats["t_preprocess"] += time.perf_counter() - t0
        if feat is None:
            return None

        self._proc.append(feat)
        if len(self._proc) < self.seq_len:
            return None

        if self.bg_mode == "concat":
            x_np = np.concatenate(
                [self._median.astype(np.float32) / 255.0] + list(self._proc), axis=0
            )
        else:
            x_np = np.concatenate(list(self._proc), axis=0)

        ort_inputs = {self.input_name: x_np[None, ...]}
        t1 = time.perf_counter()
        outputs = self.session.run(self.output_names, ort_inputs)
        self.stats["t_forward"] += time.perf_counter() - t1

        t2 = time.perf_counter()
        y = outputs[0][0]  # Remove batch dimension
        ids = list(self._fidq)

        for j, fid in enumerate(ids):
            w = float(self._ens_w[self.seq_len - 1 - j]) if self.eval_mode != "nonoverlap" else 1.0
            if fid not in self._acc_sum:
                self._acc_sum[fid] = (y[j] * w).astype(np.float32)
                self._acc_w[fid] = w
            else:
                self._acc_sum[fid] += (y[j] * w).astype(np.float32)
                self._acc_w[fid] += w

        out_id = frame_id - (self.seq_len - 1)
        out = None
        if out_id in self._acc_sum:
            heat = self._acc_sum[out_id] / max(1e-6, self._acc_w[out_id])
            x_out, y_out, v_out = self._heatmap_to_xy(heat)
            del self._acc_sum[out_id]
            del self._acc_w[out_id]
            out = {
                "Frame": int(out_id),
                "X": int(x_out),
                "Y": int(y_out),
                "Visibility": int(v_out),
            }
            self.stats["outputs"] += 1

        self.stats["t_post"] += time.perf_counter() - t2
        return out

    def flush(self) -> list[dict[str, Any]]:
        outs = []
        for fid in sorted(self._acc_sum.keys()):
            heat = self._acc_sum[fid] / max(1e-6, self._acc_w[fid])
            x_out, y_out, v_out = self._heatmap_to_xy(heat)
            outs.append(
                {
                    "Frame": int(fid),
                    "X": int(x_out),
                    "Y": int(y_out),
                    "Visibility": int(v_out),
                }
            )
        self._acc_sum.clear()
        self._acc_w.clear()
        self._proc.clear()
        self._fidq.clear()
        return outs


class InpaintModule:
    """
    Streaming InpaintNet for ONNX.
    Matches the behavior of PT's InpaintModule.
    """

    def __init__(
        self,
        model_path: str,
        seq_len: int = 16,
        batch_size: int = 4,
        device: Any = None,
        img_scaler: Any = None,
    ):
        self.sess = ort.InferenceSession(model_path)
        self.seq_len = seq_len
        self.batch_size = batch_size
        self.device = device
        self.img_scaler = img_scaler
        self._coords = []
        self._mask = []
        self._frame_ids = []

    def push(self, pred: dict, img_scaler=None) -> dict | None:
        if img_scaler is not None:
            self.img_scaler = img_scaler

        fid = int(pred["Frame"])
        x = int(pred["X"])
        y = int(pred["Y"])
        vis = int(pred["Visibility"])

        sx, sy = self.img_scaler

        # Normalize
        if vis == 1 and x > 0 and y > 0:
            cx = float(x) / (float(WIDTH) * sx)
            cy = float(y) / (float(HEIGHT) * sy)
            cx = min(max(cx, 0.0), 1.0)
            cy = min(max(cy, 0.0), 1.0)
            m = 0.0
        else:
            cx, cy = 0.0, 0.0
            m = 1.0

        self._coords.append([cx, cy])
        self._mask.append([m])
        self._frame_ids.append(fid)

        if len(self._coords) < self.seq_len:
            return None

        # Prepare batch (size 4 as per model requirement)
        batch_coords = np.zeros((self.batch_size, self.seq_len, 2), dtype=np.float32)
        batch_mask = np.zeros((self.batch_size, self.seq_len, 1), dtype=np.float32)

        batch_coords[0] = np.array(self._coords, dtype=np.float32)
        batch_mask[0] = np.array(self._mask, dtype=np.float32)

        outputs = self.sess.run(None, {"coords": batch_coords, "mask": batch_mask})
        out_batch = outputs[0]  # (4, 16, 2)

        # PT logic: out = out * mask + coor * (1.0 - mask)
        # We only care about the first frame in the current window for streaming output
        out_coor = out_batch[0, 0] * batch_mask[0, 0] + batch_coords[0, 0] * (
            1.0 - batch_mask[0, 0]
        )

        out_fid = self._frame_ids.pop(0)
        self._coords.pop(0)
        self._mask.pop(0)

        px = int(out_coor[0] * float(WIDTH) * sx)
        py = int(out_coor[1] * float(HEIGHT) * sy)
        v = 0 if (px <= 0 or py <= 0) else 1

        return {"Frame": out_fid, "X": px, "Y": py, "Visibility": v}

    def flush(self) -> list[dict]:
        results = []
        while self._frame_ids:
            fid = self._frame_ids.pop(0)
            c = self._coords.pop(0)
            m = self._mask.pop(0)
            sx, sy = self.img_scaler

            if m[0] >= 0.5:
                px, py, v = 0, 0, 0
            else:
                px = int(c[0] * float(WIDTH) * sx)
                py = int(c[1] * float(HEIGHT) * sy)
                v = 0 if (px <= 0 or py <= 0) else 1
            results.append({"Frame": fid, "X": px, "Y": py, "Visibility": v})
        return results
