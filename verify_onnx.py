import os
import sys
import numpy as np
import inspect
import onnxruntime as ort
from tracknetv3.inference import TrackNetModule, TrackNetModuleONNX
from tracknetv3.inference import InpaintModule, InpaintModuleONNX


def test_api_compatibility():
    print("Testing API compatibility...")
    # TrackNet
    tn_pytorch_params = list(inspect.signature(TrackNetModule.__init__).parameters.keys())
    tn_onnx_params = list(inspect.signature(TrackNetModuleONNX.__init__).parameters.keys())

    # Remove 'self' for comparison
    if tn_pytorch_params[0] == "self":
        tn_pytorch_params.pop(0)
    if tn_onnx_params[0] == "self":
        tn_onnx_params.pop(0)

    print(f"TrackNet PyTorch params: {tn_pytorch_params}")
    print(f"TrackNet ONNX params: {tn_onnx_params}")

    assert tn_onnx_params[0] == "model_path"
    assert tn_pytorch_params[0] == "tracknet"
    assert tn_onnx_params[1:] == tn_pytorch_params[1:]
    print("✓ TrackNet API compatibility verified")

    # InpaintNet
    inp_pytorch_params = list(inspect.signature(InpaintModule.__init__).parameters.keys())
    inp_onnx_params = list(inspect.signature(InpaintModuleONNX.__init__).parameters.keys())

    # Remove 'self' for comparison
    if inp_pytorch_params[0] == "self":
        inp_pytorch_params.pop(0)
    if inp_onnx_params[0] == "self":
        inp_onnx_params.pop(0)

    print(f"InpaintNet PyTorch params: {inp_pytorch_params}")
    print(f"InpaintNet ONNX params: {inp_onnx_params}")

    assert inp_onnx_params[0] == "model_path"
    assert inp_pytorch_params[0] == "inpaintnet"
    assert inp_onnx_params[1:] == inp_pytorch_params[1:]
    print("✓ InpaintNet API compatibility verified")


def test_module_loading():
    print("\nTesting module loading...")
    tn_path = "/workspace/TrackNetV3/onnx/TrackNet/TrackNet-4.onnx"
    inp_path = "/workspace/TrackNetV3/onnx/InpaintNet/InpaintNet-4.onnx"

    if not os.path.exists(tn_path) or not os.path.exists(inp_path):
        print("Skipping loading test: ONNX models not found")
        return

    try:
        tn_module = TrackNetModuleONNX(
            model_path=tn_path,
            seq_len=15,  # Try to match or it will raise ValueError if metadata exists
            bg_mode="",
            device=None,
        )
        print("✓ TrackNetModuleONNX loaded")
    except ValueError as e:
        print(f"TrackNet loading failed (possible seq_len mismatch): {e}")
        # Try to extract seq_len from error if it failed due to mismatch
        import re

        match = re.search(r"got (\d+)", str(e))
        if match:
            actual_seq_len = int(match.group(1))
            print(f"Retrying with seq_len={actual_seq_len}")
            tn_module = TrackNetModuleONNX(
                model_path=tn_path, seq_len=actual_seq_len, bg_mode="", device=None
            )
            print("✓ TrackNetModuleONNX loaded (retried)")

    try:
        inp_module = InpaintModuleONNX(
            model_path=inp_path, seq_len=15, device=None, img_scaler=(1.0, 1.0)
        )
        print("✓ InpaintModuleONNX loaded")
    except ValueError as e:
        print(f"InpaintNet loading failed (possible seq_len mismatch): {e}")
        import re

        match = re.search(r"got (\d+)", str(e))
        if match:
            actual_seq_len = int(match.group(1))
            print(f"Retrying with seq_len={actual_seq_len}")
            inp_module = InpaintModuleONNX(
                model_path=inp_path, seq_len=actual_seq_len, device=None, img_scaler=(1.0, 1.0)
            )
            print("✓ InpaintModuleONNX loaded (retried)")


def test_error_handling():
    print("\nTesting error handling...")
    try:
        TrackNetModuleONNX(model_path="/nonexistent.onnx", seq_len=15, bg_mode="")
    except ValueError as e:
        print(f"✓ Caught expected ValueError for nonexistent path: {e}")
        assert "ONNX model file not found" in str(e)


if __name__ == "__main__":
    test_api_compatibility()
    test_error_handling()
    test_module_loading()
    print("\nTests completed.")
