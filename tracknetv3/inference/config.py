from dataclasses import dataclass
from typing import Literal

EvalMode = Literal["nonoverlap", "average", "weight"]


@dataclass
class TrackNetConfig:
    tracknet_ckpt: str
    inpaintnet_ckpt: str = ""  # optional
    batch_size: int = 16
    eval_mode: EvalMode = "weight"
    large_video: bool = False
    max_sample_num: int = 1800
    video_range: tuple[int, int] | None = None  # (start_sec, end_sec) or None
    num_workers_cap: int = 16
