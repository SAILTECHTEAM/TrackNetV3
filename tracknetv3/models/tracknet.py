"""TrackNet model for shuttlecock detection."""

import torch
import torch.nn as nn

from tracknetv3.models.blocks import Double2DConv, Triple2DConv


class TrackNet(nn.Module):
    """U-Net based model for shuttlecock heatmap generation."""

    def __init__(self, in_dim, out_dim):
        super(TrackNet, self).__init__()
        self.down_block_1 = Double2DConv(in_dim, 64)
        self.down_block_2 = Double2DConv(64, 128)
        self.down_block_3 = Triple2DConv(128, 256)
        self.bottleneck = Triple2DConv(256, 512)
        self.up_block_1 = Triple2DConv(768, 256)
        self.up_block_2 = Double2DConv(384, 128)
        self.up_block_3 = Double2DConv(192, 64)
        self.predictor = nn.Conv2d(64, out_dim, (1, 1))
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x1 = self.down_block_1(x)  # (N,   64,  288,   512)
        x = nn.MaxPool2d((2, 2), stride=(2, 2))(x1)  # (N,   64,  144,   256)
        x2 = self.down_block_2(x)  # (N,  128,  144,   256)
        x = nn.MaxPool2d((2, 2), stride=(2, 2))(x2)  # (N,  128,   72,   128)
        x3 = self.down_block_3(x)  # (N,  256,   72,   128)
        x = nn.MaxPool2d((2, 2), stride=(2, 2))(x3)  # (N,  256,   36,    64)
        x = self.bottleneck(x)  # (N,  512,   36,    64)
        x = torch.cat(
            [nn.Upsample(scale_factor=2)(x), x3], dim=1
        )  # (N,  768,   72,   128)
        x = self.up_block_1(x)  # (N,  256,   72,   128)
        x = torch.cat(
            [nn.Upsample(scale_factor=2)(x), x2], dim=1
        )  # (N,  384,  144,   256)
        x = self.up_block_2(x)  # (N,  128,  144,   256)
        x = torch.cat(
            [nn.Upsample(scale_factor=2)(x), x1], dim=1
        )  # (N,  192,  288,   512)
        x = self.up_block_3(x)  # (N,   64,  288,   512)
        x = self.predictor(x)  # (N,    3,  288,   512)
        x = self.sigmoid(x)  # (N,    3,  288,   512)
        return x
