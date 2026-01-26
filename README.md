# TrackNetV3: Enhancing ShuttleCock Tracking with Augmentations and Trajectory Rectification

TrackNetV3 is an advanced shuttlecock tracking system that leverages deep learning to accurately track shuttlecock trajectories in badminton videos. Built on the foundation of TrackNetV2, TrackNetV3 introduces streaming inference capabilities for real-time applications, supporting Offline, Online, and Live processing modes. The system consists of two main modules: TrackNet for shuttlecock detection and InpaintNet for trajectory rectification.

[[paper](https://dl.acm.org/doi/10.1145/3595916.3626370)]

## Key Features

- **High Accuracy Tracking**: Achieves 97.51% accuracy in shuttlecock detection and tracking
- **Streaming Inference Modes**:
  - **Offline Mode**: Process entire videos at once for batch analysis
  - **Online Mode**: Real-time frame-by-frame processing with trajectory rectification
  - **Live Mode**: Direct processing from cameras or RTSP streams with immediate visualization
- **Trajectory Rectification**: InpaintNet module corrects shuttlecock paths for smoother trajectories
- **Multiple Input Sources**: Support for video files, live cameras, and RTSP streams
- **Practical Applications**: Ideal for badminton coaching, automated scoring systems, and sports broadcasting
- **Error Analysis Dashboard**: Interactive web interface for analyzing tracking performance

## Performance

| Model      | Accuracy | Precision | Recall | F1   | FPS |
|------------|----------|-----------|--------|------|-----|
| TrackNet   | 94.12%   | 93.83%    | 94.41% | 94.12% | 28.6 |
| TrackNetV2 | 96.82%   | 96.43%    | 97.21% | 96.82% | 29.8 |
| TrackNetV3 | 97.51%   | 96.83%    | 97.16% | 97.00% | 30.0 |

![Performance Comparison](figure/Comparison.png)

## Installation

### Requirements
- Ubuntu 18.04+
- Python 3.8+
- PyTorch 1.10.0+

### Setup
```bash
git clone https://github.com/xxxx/TrackNetV3.git
cd TrackNetV3
pip install -r requirements.txt
```

## Model Checkpoints

TrackNetV3 requires pre-trained checkpoints for both TrackNet and InpaintNet models. You can download them from the following link:

**Download Checkpoints:** [TrackNetV3_ckpts.zip](https://drive.google.com/file/d/1CfzE87a0f6LhBp0kniSl1-89zaLCZ8cA/view?usp=sharing)

After downloading, unzip the file and place the checkpoint files in the `ckpts/` directory:

```bash
# Download the checkpoints
wget --no-check-certificate 'https://drive.google.com/uc?export=download&id=1CfzE87a0f6LhBp0kniSl1-89zaLCZ8cA' -O TrackNetV3_ckpts.zip

# Unzip and organize
unzip TrackNetV3_ckpts.zip
mkdir -p ckpts
mv TrackNetV3_ckpts/* ckpts/
```

The checkpoint directory should contain:
- `ckpts/TrackNet_best.pt` - Pre-trained TrackNet model
- `ckpts/InpaintNet_best.pt` - Pre-trained InpaintNet model

## Quick Start

### Offline Mode
Process an entire video file for complete trajectory analysis:

```bash
python demo_offline.py --input_video path/to/video.mp4 --output_csv results.csv --output_video output.mp4
```

### Online Mode
Stream frame-by-frame processing with trajectory rectification:

```bash
python demo_online.py --input_video path/to/video.mp4 --output_csv results.csv --output_video output.mp4
```

### Live Mode
Real-time processing from camera or RTSP stream:

```bash
# From webcam
python demo_live.py --input 0

# From RTSP stream
python demo_live.py --input rtsp://your-stream-url

# From video file (for testing)
python demo_live.py --input path/to/video.mp4
```

## Usage

### Inference Classes

TrackNetV3 provides three main inference classes for different processing needs:

- **TrackNetInfer**: Designed for offline processing of complete videos. This class processes entire video files at once, making it suitable for batch processing and complete trajectory analysis.
  
- **TrackNetModule**: Optimized for streaming applications. This class handles frame-by-frame inference while maintaining state across frames, enabling real-time processing.

- **InpaintModule**: Trajectory rectification module for streaming scenarios. Works alongside TrackNetModule to correct and smooth shuttlecock paths in real-time.

### Demo Scripts

#### Offline Demo (`demo_offline.py`)
Processes entire videos using the TrackNetInfer class.

**Arguments:**
- `--input_video`: Path to input video file
- `--output_csv`: Path to save tracking results as CSV
- `--output_video`: Path to save annotated output video

**Example:**
```bash
python demo_offline.py --input_video badminton_match.mp4 --output_csv tracking_results.csv --output_video annotated_match.mp4
```

#### Online Demo (`demo_online.py`)
Demonstrates streaming frame-by-frame processing with trajectory rectification using TrackNetModule and InpaintModule.

**Arguments:**
- `--input_video`: Path to input video file
- `--output_csv`: Path to save tracking results as CSV
- `--output_video`: Path to save annotated output video

**Example:**
```bash
python demo_online.py --input_video live_feed.mp4 --output_csv online_results.csv --output_video corrected_feed.mp4
```

#### Live Demo (`demo_live.py`)
Real-time processing from various sources using TrackNetModule.

**Arguments:**
- `--input`: Input source (0 for webcam, RTSP URL, or video file path)

**Examples:**
```bash
# Webcam input
python demo_live.py --input 0

# RTSP stream
python demo_live.py --input rtsp://192.168.1.100:554/stream

# Video file (for testing live processing)
python demo_live.py --input test_video.mp4
```

## Dataset Preparation

### Download Dataset
The TrackNetV3 dataset is available from the original TrackNetV2 repository:

```bash
# Download the badminton dataset
wget https://github.com/xxxx/dataset/raw/master/badminton_dataset.zip
unzip badminton_dataset.zip
```

### Dataset Structure
The dataset should be organized as follows:

```
dataset/
├── train/
│   ├── video_001.mp4
│   ├── video_001.csv
│   ├── video_002.mp4
│   ├── video_002.csv
│   └── ...
├── val/
│   ├── video_003.mp4
│   ├── video_003.csv
│   └── ...
└── test/
    ├── video_004.mp4
    ├── video_004.csv
    └── ...
```

Each video file should have a corresponding CSV file containing the shuttlecock positions with columns: `frame`, `x`, `y`, `width`, `height`, `label`.

### Preprocessing
Preprocess the dataset to generate training data:

```bash
python scripts/preprocess.py \
    --dataset_path /path/to/dataset \
    --output_path /path/to/preprocessed \
    --seq_len 8 \
    --bg_mode subtract
```

**Preprocessing Arguments:**
- `--dataset_path`: Path to the raw dataset
- `--output_path`: Path to save preprocessed data
- `--seq_len`: Length of input sequence (default: 8)
- `--bg_mode`: Background mode for preprocessing
  - `""`: RGB input (L x 3 channels)
  - `"subtract"`: Difference frame (L x 1 channel) - **Recommended**
  - `"subtract_concat"`: RGB + Difference frame (L x 4 channels)
  - `"concat"`: RGB with extra frame (L+1 x 3 channels)
- `--num_workers`: Number of workers for data loading (default: 4)

## Training

### Training TrackNet
Train the shuttlecock detection model:

```bash
python scripts/train.py --model_name TrackNet --seq_len 8 --epochs 50 --batch_size 8 --save_dir exp_tracknet
```

**Key Arguments:**
- `--model_name`: Model type (`TrackNet` or `InpaintNet`)
- `--seq_len`: Length of input sequence (default: 8)
- `--epochs`: Number of training epochs
- `--batch_size`: Batch size for training
- `--save_dir`: Directory to save checkpoints and logs
- `--bg_mode`: Background mode for TrackNet
  - `""`: RGB input (L x 3 channels)
  - `"subtract"`: Difference frame (L x 1 channel) - **Recommended**
  - `"subtract_concat"`: RGB + Difference frame (L x 4 channels)
  - `"concat"`: RGB with extra frame (L+1 x 3 channels)
- `--alpha`: Alpha for sample mixup (default: -1, no mixup)
- `--lr_scheduler`: Learning rate scheduler (`StepLR` or empty)
- `--resume_training`: Resume training from checkpoint

### Training InpaintNet
Train the trajectory rectification model:

```bash
python scripts/train.py --model_name InpaintNet --seq_len 8 --epochs 30 --batch_size 4 --save_dir exp_inpaintnet
```

**Key Arguments:**
- `--mask_ratio`: Ratio of random mask during training InpaintNet (default: 0.3)
- `--tolerance`: Difference tolerance for evaluation (default: 4)

### Resuming Training
Resume training from a checkpoint:

```bash
python scripts/train.py --model_name TrackNet --save_dir exp_tracknet --resume_training --epochs 100
```

### TensorBoard Visualization
Monitor training progress with TensorBoard:

```bash
tensorboard --logdir exp_tracknet/logs
# View at http://localhost:6006/
```

## Evaluation

### Testing TrackNet
Evaluate TrackNet model on test set:

```bash
python scripts/test.py \
    --tracknet_file ckpts/TrackNet_best.pt \
    --split test \
    --eval_mode weight \
    --tolerance 4 \
    --save_dir output_tracknet
```

### Testing TrackNetV3 (TrackNet + InpaintNet)
Evaluate the full TrackNetV3 pipeline:

```bash
python scripts/test.py \
    --tracknet_file ckpts/TrackNet_best.pt \
    --inpaintnet_file ckpts/InpaintNet_best.pt \
    --split test \
    --eval_mode weight \
    --tolerance 4 \
    --save_dir output_tracknetv3
```

### Testing on Video File
Evaluate on a specific video file:

```bash
python scripts/test.py \
    --tracknet_file ckpts/TrackNet_best.pt \
    --inpaintnet_file ckpts/InpaintNet_best.pt \
    --video_file /path/to/video.mp4 \
    --save_dir output_video
```

### Key Evaluation Arguments
- `--tracknet_file`: Path to TrackNet checkpoint
- `--inpaintnet_file`: Path to InpaintNet checkpoint (optional)
- `--split`: Dataset split (`train`, `val`, or `test`)
- `--eval_mode`: Temporal ensemble mode
  - `weight`: Positional weight (default)
  - `average`: Uniform weight
  - `nonoverlap`: No temporal ensemble
- `--tolerance`: Tolerance for FP1 evaluation (default: 4)
- `--linear_interp`: Use linear interpolation for trajectory correction
- `--output_pred`: Output detailed prediction results for error analysis
- `--output_bbox`: Output COCO format bbox for mAP evaluation
- `--verbose`: Show progress bar
- `--debug`: Run with debug mode (limited samples)

## Error Analysis

TrackNetV3 provides an interactive error analysis dashboard built with Dash.

To run the error analysis interface:

```bash
python error_analysis.py
```

This will start a web server. Open your browser and navigate to `http://127.0.0.1:8050/` to access the dashboard.

The dashboard allows you to:
- Visualize tracking results
- Analyze prediction errors
- Compare different models
- Inspect individual frames

## References

- [TrackNetV2 Paper](https://arxiv.org/abs/xxxx)
- [Dataset](https://github.com/xxxx/dataset)
- [Labeling Tool](https://github.com/xxxx/labeling)