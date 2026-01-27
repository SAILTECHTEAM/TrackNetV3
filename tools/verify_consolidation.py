from __future__ import annotations

import sys

import numpy as np

from tracknet.core.config.constants import HEIGHT, WIDTH
from tracknet.core.inference.base import BaseInpaintModule, BaseTrackNetModule


class PTStubTrackNet(BaseTrackNetModule):
    def _forward(self, x_np: np.ndarray) -> np.ndarray:
        # Deterministic heatmaps derived from input sum; same impl used by ONNXStub.
        s = float(x_np.sum())
        y = np.zeros((self.seq_len, HEIGHT, WIDTH), dtype=np.float32)
        for j in range(self.seq_len):
            # Pick a stable argmax location.
            x = int((j + int(s)) % WIDTH)
            yj = int((j + 1 + int(s)) % HEIGHT)
            y[j, yj, x] = 1.0
        return y


class ONNXStubTrackNet(PTStubTrackNet):
    pass


class PTStubInpaint(BaseInpaintModule):
    def _forward(self, coords: np.ndarray, mask: np.ndarray) -> np.ndarray:
        # Output constant center; blend handled by base.
        out = np.zeros_like(coords, dtype=np.float32)
        out[..., 0] = 0.5
        out[..., 1] = 0.5
        return out


class ONNXStubInpaint(PTStubInpaint):
    pass


def _max_abs_diff(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.max(np.abs(np.asarray(a) - np.asarray(b))))


def main() -> int:
    rng = np.random.default_rng(0)

    frames = [rng.integers(0, 256, size=(720, 1280, 3), dtype=np.uint8) for _ in range(10)]

    pt = PTStubTrackNet(seq_len=3, bg_mode="", eval_mode="weight")
    onnx = ONNXStubTrackNet(seq_len=3, bg_mode="", eval_mode="weight")

    pt_outs: list[dict] = []
    onnx_outs: list[dict] = []
    max_dx = 0.0
    max_dy = 0.0
    max_dpred = 0.0

    for i, f in enumerate(frames):
        po = pt.push(f, frame_id=i)
        oo = onnx.push(f, frame_id=i)

        if pt._last_x_np is not None and onnx._last_x_np is not None:
            max_dx = max(max_dx, _max_abs_diff(pt._last_x_np, onnx._last_x_np))
        if pt._last_y_np is not None and onnx._last_y_np is not None:
            max_dy = max(max_dy, _max_abs_diff(pt._last_y_np, onnx._last_y_np))

        if po is not None:
            pt_outs.append(po)
        if oo is not None:
            onnx_outs.append(oo)

    pt_outs.extend(pt.flush())
    onnx_outs.extend(onnx.flush())

    if len(pt_outs) != len(onnx_outs):
        print(f"Parity check: FAIL (different output counts: {len(pt_outs)} vs {len(onnx_outs)})")
        return 1

    for a, b in zip(pt_outs, onnx_outs, strict=True):
        for k in ("Frame", "X", "Y", "Visibility"):
            max_dpred = max(max_dpred, abs(float(a[k]) - float(b[k])))

    atol = 1e-6
    ok = max_dx <= atol and max_dy <= atol and max_dpred <= atol

    print(f"max|dx|={max_dx:.3e} max|dy|={max_dy:.3e} max|dpred|={max_dpred:.3e} atol={atol:.1e}")
    print("Parity check: PASS" if ok else "Parity check: FAIL")

    # Also validate inpaint stubs parity.
    pt_ip = PTStubInpaint(seq_len=4, img_scaler=pt.img_scaler)
    onnx_ip = ONNXStubInpaint(seq_len=4, img_scaler=onnx.img_scaler)
    pt_ip_outs: list[dict] = []
    onnx_ip_outs: list[dict] = []

    for p, o in zip(pt_outs, onnx_outs, strict=True):
        a = pt_ip.push(p)
        b = onnx_ip.push(o)
        if a is not None:
            pt_ip_outs.append(a)
        if b is not None:
            onnx_ip_outs.append(b)

    pt_ip_outs.extend(pt_ip.flush())
    onnx_ip_outs.extend(onnx_ip.flush())

    if len(pt_ip_outs) != len(onnx_ip_outs):
        print("Inpaint parity: FAIL (different output counts)")
        return 1
    max_dip = 0.0
    for a, b in zip(pt_ip_outs, onnx_ip_outs, strict=True):
        for k in ("Frame", "X", "Y", "Visibility"):
            max_dip = max(max_dip, abs(float(a[k]) - float(b[k])))
    if max_dip > atol:
        print(f"Inpaint parity: FAIL (max diff {max_dip})")
        return 1
    print("Inpaint parity: PASS")

    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
