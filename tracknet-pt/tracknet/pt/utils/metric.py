"""Metric functions for TrackNetV3."""

import torch
import torch.nn as nn
import torch.nn.functional as F


def get_metric(TP, TN, FP1, FP2, FN):
    """Calculate evaluation metrics from confusion matrix.

    Args:
        TP (int): True Positives
        TN (int): True Negatives
        FP1 (int): False Positives Type 1 (wrong position)
        FP2 (int): False Positives Type 2 (false detection)
        FN (int): False Negatives

    Returns:
        Tuple[float, float, float, float]: accuracy, precision, recall, f1, miss_rate
    """
    # Accuracy: (TP + TN) / (TP + TN + FP1 + FP2 + FN)
    total = TP + TN + FP1 + FP2 + FN
    accuracy = (TP + TN) / total if total > 0 else 0.0

    # Precision: TP / (TP + FP1 + FP2)
    precision = TP / (TP + FP1 + FP2) if (TP + FP1 + FP2) > 0 else 0.0

    # Recall: TP / (TP + FN)
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0.0

    # F1 Score: 2 * (precision * recall) / (precision + recall)
    if precision + recall > 0:
        f1 = 2 * (precision * recall) / (precision + recall)
    else:
        f1 = 0.0

    # Miss Rate: FN / (TP + FN)
    miss_rate = FN / (TP + FN) if (TP + FN) > 0 else 0.0

    return accuracy, precision, recall, f1, miss_rate


class WBCELoss(nn.Module):
    """Weighted Binary Cross Entropy Loss for heatmap prediction.

    This loss function is commonly used in object detection and segmentation tasks
    where the positive (object) and negative (background) regions are imbalanced.
    The weight parameter can be used to balance the loss between positive and negative samples.
    """

    def __init__(self, pos_weight=1.0):
        """Initialize WBCELoss.

        Args:
            pos_weight (float): Weight for positive class. Default is 1.0.
        """
        super().__init__()
        self.pos_weight = torch.tensor([pos_weight])

    def forward(self, pred, target):
        """Calculate weighted binary cross entropy loss.

        Args:
            pred (torch.Tensor): Predicted heatmaps with shape (N, L, H, W)
            target (torch.Tensor): Ground truth heatmaps with shape (N, L, H, W)

        Returns:
            torch.Tensor: Scalar loss value
        """
        # Ensure predictions are in valid range for BCE
        pred = torch.clamp(pred, min=1e-7, max=1 - 1e-7)

        # Calculate binary cross entropy with logits for numerical stability
        loss = F.binary_cross_entropy(pred, target, reduction="mean")

        return loss
