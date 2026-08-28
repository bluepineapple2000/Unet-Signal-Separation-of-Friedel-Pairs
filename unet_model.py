import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint

from unet_parts import DoubleConv, Down, OutConv, Up


class UNet(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=False, base_features=64):
        super().__init__()
        if base_features <= 0:
            raise ValueError(f"base_features must be positive, got {base_features}")

        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear
        self.base_features = base_features
        self.gradient_checkpointing = False

        factor = 2 if bilinear else 1
        self.inc = DoubleConv(n_channels, base_features)
        self.down1 = Down(base_features, base_features * 2)
        self.down2 = Down(base_features * 2, base_features * 4)
        self.down3 = Down(base_features * 4, base_features * 8)
        self.down4 = Down(base_features * 8, base_features * 16 // factor)
        self.up1 = Up(base_features * 16, base_features * 8 // factor, bilinear)
        self.up2 = Up(base_features * 8, base_features * 4 // factor, bilinear)
        self.up3 = Up(base_features * 4, base_features * 2 // factor, bilinear)
        self.up4 = Up(base_features * 2, base_features, bilinear)
        self.outc = OutConv(base_features, n_classes)

    def _maybe_checkpoint(self, module, *inputs):
        if self.gradient_checkpointing and self.training:
            return checkpoint(module, *inputs, use_reentrant=False)
        return module(*inputs)

    def forward(self, x):
        x1 = self._maybe_checkpoint(self.inc, x)
        x2 = self._maybe_checkpoint(self.down1, x1)
        x3 = self._maybe_checkpoint(self.down2, x2)
        x4 = self._maybe_checkpoint(self.down3, x3)
        x5 = self._maybe_checkpoint(self.down4, x4)
        x = self._maybe_checkpoint(self.up1, x5, x4)
        x = self._maybe_checkpoint(self.up2, x, x3)
        x = self._maybe_checkpoint(self.up3, x, x2)
        x = self._maybe_checkpoint(self.up4, x, x1)
        return self.outc(x)

    def use_checkpointing(self):
        self.gradient_checkpointing = True
