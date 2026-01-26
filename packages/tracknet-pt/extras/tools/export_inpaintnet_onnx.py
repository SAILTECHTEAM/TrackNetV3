import argparse
import os
import torch

from tracknet.pt.models.factory import get_model


def main(args):
    if not os.path.exists(args.checkpoint):
        raise FileNotFoundError(f"Checkpoint {args.checkpoint} not found.")

    print(f"Loading checkpoint from {args.checkpoint}...")
    checkpoint = torch.load(args.checkpoint, map_location="cpu")

    if "param_dict" not in checkpoint:
        raise ValueError("'param_dict' not found in checkpoint. Cannot extract required metadata.")

    param_dict = checkpoint["param_dict"]

    # InpaintNet needs seq_len for input shape
    if "seq_len" not in param_dict:
        raise ValueError("'seq_len' not found in checkpoint's param_dict.")

    model_name = param_dict.get("model_name", "InpaintNet")
    seq_len = int(param_dict["seq_len"])

    print(f"Detected model: {model_name}, seq_len: {seq_len}")

    # Instantiate model
    model = get_model(model_name)
    if "model" in checkpoint:
        model.load_state_dict(checkpoint["model"])
    else:
        raise ValueError("'model' state_dict not found in checkpoint.")

    model.eval()

    # InpaintNet takes two inputs: coordinates and mask
    batch_size = args.batch_size
    dummy_coords = torch.randn(batch_size, seq_len, 2)
    dummy_mask = torch.randn(batch_size, seq_len, 1)

    input_names = ["coords", "mask"]
    output_names = ["output"]

    dynamic_axes = None
    if args.dynamic_batch:
        dynamic_axes = {
            "coords": {0: "batch_size"},
            "mask": {0: "batch_size"},
            "output": {0: "batch_size"},
        }

    # Prepare metadata
    metadata = {
        "model_name": model_name,
        "seq_len": str(seq_len),
        "batch_mode": "dynamic" if args.dynamic_batch else "fixed",
        "batch_size": str(batch_size),
    }

    # If the provided output is a directory (or ends with a path separator),
    # construct a default filename inside it: TrackNet/InpaintNet-{bs|dynamic}.onnx
    batch_label = "dynamic" if args.dynamic_batch else str(args.batch_size)
    output_path = args.output
    if os.path.isdir(output_path) or output_path.endswith(os.path.sep):
        # Use InpaintNet-specific directory/filename when exporting from the InpaintNet script
        output_path = os.path.join(output_path, f"InpaintNet/{model_name}-{batch_label}.onnx")
    # Ensure parent directories exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    args.output = output_path

    print(f"Exporting InpaintNet to {args.output}...")
    torch.onnx.export(
        model,
        (dummy_coords, dummy_mask),
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
    parser = argparse.ArgumentParser(description="Export InpaintNet model to ONNX")
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
