"""TrackNetV3: Badminton shuttlecock tracking with deep learning."""

__version__ = "1.0.0"

# Public API - Inference
# Public API - Config
from tracknetv3.config.constants import COOR_TH, HEIGHT, WIDTH

# Public API - Datasets
from tracknetv3.datasets import Shuttlecock_Trajectory_Dataset, Video_IterableDataset
from tracknetv3.inference import InpaintModule, TrackNetInfer, TrackNetModule
from tracknetv3.inference.config import EvalMode, TrackNetConfig

# Public API - Models
from tracknetv3.models import InpaintNet, TrackNet, get_model

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
