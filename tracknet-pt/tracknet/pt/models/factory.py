"""Model factory utilities.

Intentionally uses deep explicit imports (no package-level exports).
"""

from __future__ import annotations

from tracknet.pt.models.inpaintnet import InpaintNet
from tracknet.pt.models.tracknet import TrackNet


def get_model(model_name: str, seq_len: int | None = None, bg_mode: str | None = None):
    """Create model by name and configuration parameter."""

    if model_name == "TrackNet":
        if seq_len is None:
            raise ValueError("seq_len is required for TrackNet")

        bg_mode = bg_mode or ""
        if bg_mode == "subtract":
            return TrackNet(in_dim=seq_len, out_dim=seq_len)
        if bg_mode == "subtract_concat":
            return TrackNet(in_dim=seq_len * 4, out_dim=seq_len)
        if bg_mode == "concat":
            return TrackNet(in_dim=(seq_len + 1) * 3, out_dim=seq_len)
        return TrackNet(in_dim=seq_len * 3, out_dim=seq_len)

    if model_name == "InpaintNet":
        return InpaintNet()

    raise ValueError(f"Invalid model name: {model_name}")
