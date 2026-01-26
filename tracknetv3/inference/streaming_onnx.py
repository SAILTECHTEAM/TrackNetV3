from __future__ import annotations

import os
import time
from collections import deque
from typing import Any

import cv2
import numpy as np
import onnxruntime as ort

from tracknetv3.config.constants import HEIGHT, WIDTH
from tracknetv3.evaluation.ensemble import get_ensemble_weight


class TrackNetModuleONNX:
    """
    Streaming TrackNet (overlap ensemble, close to offline) using ONNX Runtime.
    - push(frame_bgr, frame_id) -> Optional[{"Frame","X","Y","Visibility"}]
    - flush() -> list of remaining tail predictions
    """

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

        self.frames = deque(maxlen=self.seq_len)
        self.frame_ids = deque(maxlen=self.seq_len)

        self.img_scaler: tuple[float, float] | None = None
        self.img_shape: tuple[int, int] | None = None

        # NumPy ensemble weight conversion
        self._ens_w = np.asarray(
            get_ensemble_weight(self.seq_len, self.eval_mode).cpu().numpy(), dtype=np.float32
        )

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
        self.frames.clear()
        self.frame_ids.clear()
        self._acc_sum.clear()
        self._acc_w.clear()
        self._warmup_rgb.clear()
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

    def _process_window(self, frames_bgr: list[np.ndarray]) -> np.ndarray | None:
        imgs = np.asarray(frames_bgr, dtype=np.uint8)[..., ::-1]

        if self.bg_mode and self._median is None:
            self._maybe_build_median(imgs[-1])
            if self._median is None:
                return None

        H, W = HEIGHT, WIDTH
        T = self.seq_len
        mode = self.bg_mode

        if mode == "subtract":
            c_per = 1
        elif mode == "subtract_concat":
            c_per = 4
        else:
            c_per = 3

        extra_c = 3 if mode == "concat" else 0
        out = np.empty((extra_c + T * c_per, H, W), dtype=np.float32)

        write_offset = 0
        if mode == "concat":
            out[0:3] = self._median.astype(np.float32)
            write_offset = 3
            med = None
        elif mode:
            med = self._median
        else:
            med = None

        for i in range(T):
            img = imgs[i]

            if mode == "subtract":
                diff = np.abs(img.astype(np.int16) - med.astype(np.int16)).sum(axis=2)
                diff = np.clip(diff, 0, 255).astype(np.uint8)
                diff_r = cv2.resize(diff, (W, H), interpolation=cv2.INTER_AREA)
                out[write_offset + i] = diff_r.astype(np.float32)

            elif mode == "subtract_concat":
                diff = np.abs(img.astype(np.int16) - med.astype(np.int16)).sum(axis=2)
                diff = np.clip(diff, 0, 255).astype(np.uint8)

                img_r = cv2.resize(img, (W, H), interpolation=cv2.INTER_AREA)
                diff_r = cv2.resize(diff, (W, H), interpolation=cv2.INTER_AREA)

                base = write_offset + i * 4
                out[base + 0] = img_r[:, :, 0].astype(np.float32)
                out[base + 1] = img_r[:, :, 1].astype(np.float32)
                out[base + 2] = img_r[:, :, 2].astype(np.float32)
                out[base + 3] = diff_r.astype(np.float32)

            else:
                img_r = cv2.resize(img, (W, H), interpolation=cv2.INTER_AREA)
                base = write_offset + i * 3
                out[base + 0] = img_r[:, :, 0].astype(np.float32)
                out[base + 1] = img_r[:, :, 1].astype(np.float32)
                out[base + 2] = img_r[:, :, 2].astype(np.float32)

        out *= 1.0 / 255.0
        return out

    def _heatmap_to_xy(self, heat: np.ndarray) -> tuple[int, int, int]:
        if heat is None or float(heat.max()) <= 0.0:
            return 0, 0, 0
        Hm, Wm = heat.shape
        idx = int(np.argmax(heat.reshape(-1)))
        y0 = idx // Wm
        x0 = idx % Wm
        sx, sy = self.img_scaler
        x = int(x0 * sx)
        y = int(y0 * sy)
        v = 0 if (x == 0 and y == 0) else 1
        return x, y, v

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

        elif self.bg_mode == "subtract_concat":
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

        else:
            img_r = (
                cv2.resize(rgb, (WIDTH, HEIGHT), interpolation=cv2.INTER_AREA).astype(np.float32)
                / 255.0
            )
            return np.moveaxis(img_r, -1, 0)

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

        # ONNX Runtime Inference
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


class InpaintModuleONNX:
    """
    Streaming InpaintNet using ONNX Runtime.
    - push(track_pred) -> Optional refined pred for one frame
    - flush() -> remaining tail
    """

    def __init__(
        self,
        model_path: str,
        seq_len: int,
        device: Any,
        img_scaler: tuple[float, float],
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

        self.input_coords = self.session.get_inputs()[0].name
        self.input_mask = self.session.get_inputs()[1].name
        self.output_names = [output.name for output in self.session.get_outputs()]

        # Validate metadata
        model_metadata = self.session.get_modelmeta().custom_metadata_map
        if "seq_len" in model_metadata:
            m_seq_len = int(model_metadata["seq_len"])
            if m_seq_len != int(seq_len):
                raise ValueError(
                    f"Model metadata mismatch: expected seq_len={seq_len}, got {m_seq_len}. Model={model_path}"
                )

        self.seq_len = int(seq_len)
        self.device = device
        self.img_scaler = img_scaler

        self._coords = deque(maxlen=self.seq_len)
        self._mask = deque(maxlen=self.seq_len)
        self._frame_ids = deque(maxlen=self.seq_len)

        self.stats = {"push_calls": 0, "outputs": 0, "t_forward": 0.0, "t_post": 0.0}
        self._count = 0

    def reset(self):
        self._coords.clear()
        self._mask.clear()
        self._frame_ids.clear()

    def push(self, pred: dict[str, Any]) -> dict[str, Any] | None:
        self.stats["push_calls"] += 1

        fid = int(pred["Frame"])
        x = int(pred["X"])
        y = int(pred["Y"])
        vis = int(pred["Visibility"])

        sx, sy = self.img_scaler

        if vis == 1 and x > 0 and y > 0:
            cx = float(x) / float(WIDTH * sx)
            cy = float(y) / float(HEIGHT * sy)
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

        coords_np = np.array(self._coords, dtype=np.float32)
        mask_np = np.array(self._mask, dtype=np.float32)

        # ONNX Runtime Inference
        ort_inputs = {self.input_coords: coords_np[None, ...], self.input_mask: mask_np[None, ...]}
        t1 = time.perf_counter()
        outputs = self.session.run(self.output_names, ort_inputs)
        self.stats["t_forward"] += time.perf_counter() - t1

        t2 = time.perf_counter()
        out = outputs[0][0]  # Remove batch dimension

        # Apply mask and coordinate blending
        out = out * mask_np + coords_np * (1.0 - mask_np)

        out_fid = int(self._frame_ids[0])
        cx_out, cy_out = out[0]

        px = int(cx_out * WIDTH * sx)
        py = int(cy_out * HEIGHT * sy)
        v = 0 if (px == 0 and py == 0) else 1

        self.stats["t_post"] += time.perf_counter() - t2
        self.stats["outputs"] += 1

        return {"Frame": out_fid, "X": px, "Y": py, "Visibility": v}

    def flush(self) -> list[dict[str, Any]]:
        outs = []
        while self._frame_ids:
            fid = self._frame_ids.popleft()
            c = self._coords.popleft()
            m = self._mask.popleft()
            if m[0] >= 0.5:
                px, py, v = 0, 0, 0
            else:
                sx, sy = self.img_scaler
                px = int(c[0] * WIDTH * sx)
                py = int(c[1] * HEIGHT * sy)
                v = 0 if (px == 0 and py == 0) else 1
            outs.append({"Frame": int(fid), "X": int(px), "Y": int(py), "Visibility": int(v)})
        return outs
