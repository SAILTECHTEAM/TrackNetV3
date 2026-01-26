"""InpaintNet model for trajectory inpainting."""

import torch
import torch.nn as nn
from tracknet.pt.models.blocks import Conv1DBlock, Double1DConv


class InpaintNet(nn.Module):
    """1D CNN based model for shuttlecock trajectory inpainting."""

    def __init__(self):
        super().__init__()
        self.down_1 = Conv1DBlock(3, 32)
        self.down_2 = Conv1DBlock(32, 64)
        self.down_3 = Conv1DBlock(64, 128)
        self.buttleneck = Double1DConv(128, 256)
        self.up_1 = Conv1DBlock(384, 128)
        self.up_2 = Conv1DBlock(192, 64)
        self.up_3 = Conv1DBlock(96, 32)
        self.predictor = nn.Conv1d(32, 2, 3, padding="same")
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, m):
        x = torch.cat([x, m], dim=2)  # (N,   L,   3)
        x = x.permute(0, 2, 1)  # (N,   3,   L)
        x1 = self.down_1(x)  # (N,  16,   L)
        x2 = self.down_2(x1)  # (N,  32,   L)
        x3 = self.down_3(x2)  # (N,  64,   L)
        x = self.buttleneck(x3)  # (N,  256,  L)
        x = torch.cat([x, x3], dim=1)  # (N,  384,  L)
        x = self.up_1(x)  # (N,  128,  L)
        x = torch.cat([x, x2], dim=1)  # (N,  192,  L)
        x = self.up_2(x)  # (N,   64,  L)
        x = torch.cat([x, x1], dim=1)  # (N,   96,  L)
        x = self.up_3(x)  # (N,   32,  L)
        x = self.predictor(x)  # (N,   2,   L)
        x = self.sigmoid(x)  # (N,   2,   L)
        x = x.permute(0, 2, 1)  # (N,   L,   2)
        return x
