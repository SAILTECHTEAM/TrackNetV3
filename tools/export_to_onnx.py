import argparse
import os

import torch

from tracknetv3.config.constants import HEIGHT, WIDTH
from tracknetv3.utils.general import get_model


def export_tracknet(model, seq_len, bg_mode, output_path, dynamic_batch=False, opset_version=14):
    """
    Exports TrackNet model to ONNX format.
    """
    # Calculate input dimensions based on bg_mode
    if bg_mode == "subtract":
        in_dim = seq_len
    elif bg_mode == "subtract_concat":
        in_dim = seq_len * 4
    elif bg_mode == "concat":
        in_dim = (seq_len + 1) * 3
    else:
        in_dim = seq_len * 3

    # Create dummy input
    dummy_input = torch.randn(1, in_dim, HEIGHT, WIDTH)

    input_names = ["input"]
    output_names = ["output"]

    dynamic_axes = None
    if dynamic_batch:
        dynamic_axes = {"input": {0: "batch_size"}, "output": {0: "batch_size"}}

    print(f"Exporting TrackNet (seq_len={seq_len}, bg_mode='{bg_mode}') to {output_path}...")
    torch.onnx.export(
        model,
        dummy_input,
        output_path,
        export_params=True,
        opset_version=opset_version,
        do_constant_folding=True,
        input_names=input_names,
        output_names=output_names,
        dynamic_axes=dynamic_axes,
    )
    print("Export complete.")


def export_inpaintnet(model, seq_len, output_path, dynamic_batch=False, opset_version=14):
    """
    Exports InpaintNet model to ONNX format.
    """
    # InpaintNet takes two inputs: coordinates and mask
    dummy_coords = torch.randn(1, seq_len, 2)
    dummy_mask = torch.randn(1, seq_len, 1)

    input_names = ["coords", "mask"]
    output_names = ["output"]

    dynamic_axes = None
    if dynamic_batch:
        dynamic_axes = {
            "coords": {0: "batch_size"},
            "mask": {0: "batch_size"},
            "output": {0: "batch_size"},
        }

    print(f"Exporting InpaintNet (seq_len={seq_len}) to {output_path}...")
    torch.onnx.export(
        model,
        (dummy_coords, dummy_mask),
        output_path,
        export_params=True,
        opset_version=opset_version,
        do_constant_folding=True,
        input_names=input_names,
        output_names=output_names,
        dynamic_axes=dynamic_axes,
    )
    print("Export complete.")


def main():
    parser = argparse.ArgumentParser(description="Export TrackNetV3 models to ONNX")
    parser.add_argument(
        "--checkpoint", type=str, required=True, help="Path to PyTorch checkpoint (.pt)"
    )
    parser.add_argument("--output_dir", type=str, default=".", help="Directory to save ONNX model")
    parser.add_argument("--dynamic-batch", action="store_true", help="Enable dynamic batch size")
    parser.add_argument("--opset", type=int, default=18, help="ONNX opset version")

    args = parser.parse_args()

    if not os.path.exists(args.checkpoint):
        print(f"Error: Checkpoint {args.checkpoint} not found.")
        return

    # Load checkpoint
    print(f"Loading checkpoint from {args.checkpoint}...")
    checkpoint = torch.load(args.checkpoint, map_location="cpu")

    if "param_dict" not in checkpoint:
        print("Error: 'param_dict' not found in checkpoint. Cannot extract metadata.")
        return

    param_dict = checkpoint["param_dict"]
    model_name = param_dict.get("model_name", "TrackNet")
    seq_len = int(param_dict.get("seq_len", 9))
    bg_mode = param_dict.get("bg_mode", "")

    print(f"Detected model: {model_name}, seq_len: {seq_len}, bg_mode: '{bg_mode}'")

    # Instantiate model
    model = get_model(model_name, seq_len, bg_mode)
    if "model" in checkpoint:
        model.load_state_dict(checkpoint["model"])
    else:
        print("Warning: 'model' state_dict not found in checkpoint. Exporting architecture only.")

    model.eval()

    # Prepare output path
    os.makedirs(args.output_dir, exist_ok=True)
    if model_name == "TrackNet":
        mode_str = bg_mode if bg_mode else "rgb"
        filename = f"tracknet_seq{seq_len}_{mode_str}.onnx"
        output_path = os.path.join(args.output_dir, filename)
        export_tracknet(model, seq_len, bg_mode, output_path, args.dynamic_batch, args.opset)
    elif model_name == "InpaintNet":
        filename = f"inpaintnet_seq{seq_len}.onnx"
        output_path = os.path.join(args.output_dir, filename)
        export_inpaintnet(model, seq_len, output_path, args.dynamic_batch, args.opset)
    else:
        print(f"Error: Unknown model name '{model_name}'")


if __name__ == "__main__":
    main()
