"""TrackNetV3: Badminton shuttlecock tracking with deep learning."""

__version__ = "1.0.0"

# Public API - Inference
from tracknetv3.inference import TrackNetInfer, TrackNetModule, InpaintModule
from tracknetv3.inference.config import TrackNetConfig, EvalMode

# Public API - Models
from tracknetv3.models import TrackNet, InpaintNet, get_model

# Public API - Datasets
from tracknetv3.datasets import Shuttlecock_Trajectory_Dataset, Video_IterableDataset

# Public API - Config
from tracknetv3.config.constants import WIDTH, HEIGHT, COOR_TH

__all__ = [
    # Inference
    "TrackNetInfer",
    "TrackNetModule",
    "InpaintModule",
    "TrackNetConfig",
    "EvalMode",
    # Models
    "TrackNet",
    "InpaintNet",
    "get_model",
    # Datasets
    "Shuttlecock_Trajectory_Dataset",
    "Video_IterableDataset",
    # Config
    "WIDTH",
    "HEIGHT",
    "COOR_TH",
]
