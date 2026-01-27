import numpy as np


def _ensemble_weights(seq_len: int, eval_mode: str) -> np.ndarray:
    """Compute temporal ensemble weights without torch dependency."""

    seq_len = int(seq_len)
    if seq_len <= 0:
        raise ValueError("seq_len must be positive")

    if eval_mode in ("nonoverlap", "average"):
        return (np.ones(seq_len, dtype=np.float32) / float(seq_len)).astype(np.float32)

    if eval_mode == "weight":
        w = np.ones(seq_len, dtype=np.float32)
        half = int(np.ceil(seq_len / 2))
        for i in range(half):
            w[i] = float(i + 1)
            w[seq_len - i - 1] = float(i + 1)
        return (w / float(w.sum())).astype(np.float32)

    raise ValueError(f"Invalid eval_mode: {eval_mode!r}")
