"""Ensemble utilities for TrackNetV3."""

import math

import torch


def get_ensemble_weight(seq_len, eval_mode):
    """Get weight for temporal ensemble.

    Args:
        seq_len (int): Length of input sequence
        eval_mode (str): Mode of temporal ensemble
            Choices:
                - 'average': Return uniform weight
                - 'weight': Return positional weight

    Returns:
        weight (torch.Tensor): Weight for temporal ensemble
    """

    if eval_mode == "average":
        weight = torch.ones(seq_len) / seq_len
    elif eval_mode == "weight":
        weight = torch.ones(seq_len)
        for i in range(math.ceil(seq_len / 2)):
            weight[i] = i + 1
            weight[seq_len - i - 1] = i + 1
        weight = weight / weight.sum()
    else:
        raise ValueError("Invalid mode")

    return weight
