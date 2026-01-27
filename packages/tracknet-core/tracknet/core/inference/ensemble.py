"""Temporal ensemble weighting (numpy-only)."""

from __future__ import annotations

import math

import numpy as np


def get_ensemble_weight(seq_len: int, eval_mode: str) -> np.ndarray:
    """Return per-position weights for temporal ensemble.

    Args:
        seq_len: Window length.
        eval_mode: One of {'average', 'weight', 'nonoverlap'}.

    Returns:
        np.ndarray of shape (seq_len,) dtype float32.
    """

    seq_len = int(seq_len)
    if seq_len <= 0:
        raise ValueError(f"seq_len must be > 0, got {seq_len}")

    if eval_mode in ("average", "nonoverlap"):
        return (np.ones(seq_len, dtype=np.float32) / float(seq_len)).astype(np.float32)

    if eval_mode == "weight":
        w = np.ones(seq_len, dtype=np.float32)
        for i in range(math.ceil(seq_len / 2)):
            w[i] = i + 1
            w[seq_len - i - 1] = i + 1
        w = w / float(w.sum())
        return w.astype(np.float32)

    raise ValueError(f"Invalid eval_mode: {eval_mode!r}")
