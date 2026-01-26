# TrackNetV3

TrackNetV3 is a production-ready shuttlecock tracking pipeline for badminton video analysis. It extends TrackNetV2 with streaming-capable inference and a trajectory-rectification module (InpaintNet) to improve continuity and reduce missed detections.

## Project Structure

This project is organized as a [uv](https://github.com/astral-sh/uv) workspace with three specialized packages:

- **`tracknet-core`**: Shared utilities, constants, and core logic used by both PyTorch and ONNX implementations.
- **`tracknet-pt`**: PyTorch-based training and inference module. Includes training scripts, dataset preprocessing, and PyTorch demos.
- **`tracknet-onnx`**: High-performance inference module using ONNX Runtime.

## Installation

This project uses `uv` for dependency management.

### 1. Install uv
If you haven't installed `uv`, follow the [official installation guide](https://github.com/astral-sh/uv).

### 2. Setup Workspace
Clone the repository and sync the environment:
```bash
git clone https://github.com/qaz812345/TrackNetV3.git
cd TrackNetV3
uv sync
```

## Model Checkpoints

**Important:** Do not sync checkpoints through Git. 

Download the official pretrained PyTorch checkpoints from the [TrackNetV3 Repository](https://github.com/qaz812345/TrackNetV3) (see Releases or Downloads section). 

Place them in the `ckpts/` directory:
- `ckpts/TrackNet_best.pt`
- `ckpts/InpaintNet_best.pt`

## Quick Start (Demos)

All scripts should be executed using `uv run`.

### Offline Mode (Batch Processing)
```bash
uv run python packages/tracknet-pt/extras/demos/demo_offline.py \
    --input_video path/to/video.mp4 \
    --output_csv results.csv \
    --output_video annotated.mp4
```

### Online Mode (Streaming + Rectification)
```bash
uv run python packages/tracknet-pt/extras/demos/demo_online.py \
    --input_video path/to/video.mp4 \
    --output_csv online_results.csv \
    --output_video corrected.mp4
```

### Live Mode (Webcam / RTSP)
```bash
# Webcam
uv run python packages/tracknet-pt/extras/demos/demo_live.py --input 0

# RTSP
uv run python packages/tracknet-pt/extras/demos/demo_live.py --input rtsp://<user>:<pass>@host:port/path
```

## Dataset Preprocessing

Generate training sequences from raw videos and CSV labels:
```bash
uv run python packages/tracknet-pt/extras/tools/preprocess.py \
    --dataset_path /path/to/dataset \
    --output_path /path/to/preprocessed \
    --seq_len 8 \
    --bg_mode subtract
```

## Training

Train TrackNet or InpaintNet models:

```bash
# TrackNet
uv run python packages/tracknet-pt/tracknet/pt/scripts/train.py \
    --model_name TrackNet --seq_len 8 --epochs 50 --batch_size 8 --save_dir exp_tracknet

# InpaintNet
uv run python packages/tracknet-pt/tracknet/pt/scripts/train.py \
    --model_name InpaintNet --seq_len 8 --epochs 30 --batch_size 4 --save_dir exp_inpaintnet
```

## ONNX Export

Export PyTorch checkpoints to ONNX for optimized inference.

Note: do not combine `--dynamic-batch` and `--bs` in the same command. Use one of the two modes:

- Dynamic-batch export (variable batch dimension): omit `--bs` and pass `--dynamic-batch`.
- Fixed-batch export (static batch size): pass `--bs <N>` and omit `--dynamic-batch`.

### Examples

Export TrackNet with a dynamic batch dimension:
```bash
uv run python packages/tracknet-pt/extras/tools/export_tracknet_onnx.py \
    --checkpoint ckpts/TrackNet_best.pt \
    --output ckpts/TrackNet.onnx \
    --dynamic-batch
```

Export TrackNet with a fixed batch size (batch size = 1):
```bash
uv run python packages/tracknet-pt/extras/tools/export_tracknet_onnx.py \
    --checkpoint ckpts/TrackNet_best.pt \
    --output ckpts/TrackNet_bs1.onnx \
    --bs 1
```

Export InpaintNet with a dynamic batch dimension:
```bash
uv run python packages/tracknet-pt/extras/tools/export_inpaintnet_onnx.py \
    --checkpoint ckpts/InpaintNet_best.pt \
    --output ckpts/InpaintNet.onnx \
    --dynamic-batch
```

Export InpaintNet with a fixed batch size (batch size = 1):
```bash
uv run python packages/tracknet-pt/extras/tools/export_inpaintnet_onnx.py \
    --checkpoint ckpts/InpaintNet_best.pt \
    --output ckpts/InpaintNet_bs1.onnx \
    --bs 1
```

## Evaluation

Benchmark models on a dataset split:
```bash
uv run python packages/tracknet-pt/tracknet/pt/scripts/test.py \
    --tracknet_file ckpts/TrackNet_best.pt \
    --inpaintnet_file ckpts/InpaintNet_best.pt \
    --split test \
    --save_dir output_eval
```
