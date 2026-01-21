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
- Python 3.6+

### Setup
```bash
git clone https://github.com/xxxx/TrackNetV3.git
cd TrackNetV3
pip install -r requirements.txt
```

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

## Training

### Dataset Preparation
Prepare the badminton video dataset following the TrackNetV2 methodology. Ensure videos are properly labeled with shuttlecock positions.

### Training TrackNet
Train the shuttlecock detection model:

```bash
python train.py --model tracknet --dataset_path /path/to/dataset --epochs 50 --batch_size 8
```

### Training InpaintNet
Train the trajectory rectification model:

```bash
python train.py --model inpaintnet --dataset_path /path/to/dataset --epochs 30 --batch_size 4
```

## Evaluation

To evaluate the model on the test set:

```bash
python test.py --model tracknet --test_set_path [path/to/test/set]
python test.py --model inpaintnet --test_set_path [path/to/test/set]
```

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