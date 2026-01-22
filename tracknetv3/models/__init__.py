"""Neural network models for TrackNetV3."""

from tracknetv3.models.blocks import (
    Conv2DBlock,
    Double2DConv,
    Triple2DConv,
    Conv1DBlock,
    Double1DConv,
)
from tracknetv3.models.tracknet import TrackNet
from tracknetv3.models.inpaintnet import InpaintNet

__all__ = [
    # Blocks
    "Conv2DBlock",
    "Double2DConv",
    "Triple2DConv",
    "Conv1DBlock",
    "Double1DConv",
    # Models
    "TrackNet",
    "InpaintNet",
]


def get_model(model_name, seq_len=None, bg_mode=None, device="cpu"):
    """
    Create model by name and configuration parameter.

    Args:
        model_name (str): Type of model to create
            Choices:
                - 'TrackNet': Return TrackNet model
                - 'InpaintNet': Return InpaintNet model
        seq_len (int, optional): Length of input sequence of TrackNet
        bg_mode (str, optional): Background mode of TrackNet
            Choices:
                - '': Return TrackNet with L x 3 input channels (RGB)
                - 'subtract': Return TrackNet with L x 1 input channel (Difference frame)
                - 'subtract_concat': Return TrackNet with L x 4 input channels (RGB + Difference frame)
                - 'concat': Return TrackNet with (L+1) x 3 input channels (RGB)
        device (str): Device to place the model on ('cpu' or 'cuda')

    Returns:
        model (torch.nn.Module): Model with specified configuration
    """
    if model_name == "TrackNet":
        if bg_mode == "subtract":
            model = TrackNet(in_dim=seq_len, out_dim=seq_len)
        elif bg_mode == "subtract_concat":
            model = TrackNet(in_dim=seq_len * 4, out_dim=seq_len)
        elif bg_mode == "concat":
            model = TrackNet(in_dim=(seq_len + 1) * 3, out_dim=seq_len)
        else:
            model = TrackNet(in_dim=seq_len * 3, out_dim=seq_len)
    elif model_name == "InpaintNet":
        model = InpaintNet()
    else:
        raise ValueError("Invalid model name.")

    model = model.to(device)
    return model
