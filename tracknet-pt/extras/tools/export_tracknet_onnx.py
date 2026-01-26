import argparse
import os
import torch
from tracknetv3.config.constants import HEIGHT, WIDTH
from tracknetv3.models import get_model


def main(args):
    if not os.path.exists(args.checkpoint):
        raise FileNotFoundError(f"Checkpoint {args.checkpoint} not found.")

    print(f"Loading checkpoint from {args.checkpoint}...")
    checkpoint = torch.load(args.checkpoint, map_location="cpu")

    if "param_dict" not in checkpoint:
        raise ValueError("'param_dict' not found in checkpoint. Cannot extract required metadata.")

    param_dict = checkpoint["param_dict"]

    # Required parameters for TrackNet
    if "seq_len" not in param_dict:
        raise ValueError("'seq_len' not found in checkpoint's param_dict.")
    if "bg_mode" not in param_dict:
        raise ValueError("'bg_mode' not found in checkpoint's param_dict.")

    model_name = param_dict.get("model_name", "TrackNet")
    seq_len = int(param_dict["seq_len"])
    bg_mode = param_dict["bg_mode"]

    print(f"Detected model: {model_name}, seq_len: {seq_len}, bg_mode: '{bg_mode}'")

    # Instantiate model
    model = get_model(model_name, seq_len, bg_mode)
    if "model" in checkpoint:
        model.load_state_dict(checkpoint["model"])
    else:
        raise ValueError("'model' state_dict not found in checkpoint.")

    model.eval()

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
    batch_size = args.batch_size
    dummy_input = torch.randn(batch_size, in_dim, HEIGHT, WIDTH)

    input_names = ["input"]
    output_names = ["output"]

    dynamic_axes = None
    if args.dynamic_batch:
        dynamic_axes = {"input": {0: "batch_size"}, "output": {0: "batch_size"}}

    # Prepare metadata
    metadata = {
        "model_name": model_name,
        "seq_len": str(seq_len),
        "bg_mode": bg_mode,
        "batch_mode": "dynamic" if args.dynamic_batch else "fixed",
        "batch_size": str(batch_size),
    }

    # If the provided output is a directory (or ends with a path separator),
    # construct a default filename inside it: TrackNet/<model_name>-{bs|dynamic}.onnx
    batch_label = "dynamic" if args.dynamic_batch else str(args.batch_size)
    output_path = args.output
    if os.path.isdir(output_path) or output_path.endswith(os.path.sep):
        output_path = os.path.join(output_path, f"TrackNet/{model_name}-{batch_label}.onnx")
    # Ensure parent directories exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    args.output = output_path

    print(f"Exporting TrackNet to {args.output}...")
    torch.onnx.export(
        model,
        dummy_input,
        args.output,
        export_params=True,
        opset_version=args.opset,
        do_constant_folding=True,
        input_names=input_names,
        output_names=output_names,
        dynamic_axes=dynamic_axes,
    )

    # Add metadata to ONNX model
    import onnx

    onnx_model = onnx.load(args.output)
    for key, value in metadata.items():
        meta = onnx_model.metadata_props.add()
        meta.key = key
        meta.value = value
    onnx.save(onnx_model, args.output)

    print("Export complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export TrackNet model to ONNX")
    parser.add_argument(
        "--checkpoint", type=str, required=True, help="Path to PyTorch checkpoint (.pt)"
    )
    parser.add_argument("--output", type=str, required=True, help="Path to save ONNX model")
    # Allow shorthand alias --bs for batch size
    parser.add_argument(
        "--batch-size",
        "--bs",
        type=int,
        default=1,
        help="Batch size for fixed batch export (alias: --bs)",
    )
    parser.add_argument("--dynamic-batch", action="store_true", help="Enable dynamic batch size")
    parser.add_argument("--opset", type=int, default=18, help="ONNX opset version")

    args = parser.parse_args()
    main(args)
