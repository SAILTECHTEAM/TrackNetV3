"""Configuration helpers.

This module is intentionally lightweight and free of heavy ML dependencies.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal


BgMode = Literal["", "subtract", "subtract_concat", "concat"]


@dataclass(frozen=True)
class DataConfig:
    """Dataset/data-layout configuration."""

    data_dir: Path
    split: str = "train"


@dataclass(frozen=True)
class InferenceConfig:
    """Inference-time configuration shared across runtimes."""

    seq_len: int = 8
    bg_mode: BgMode = ""
    eval_mode: bool = True
