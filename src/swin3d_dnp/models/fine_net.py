"""Fine network for high-resolution patch processing.

This module implements the fine-resolution network that processes
patches with fused coarse context features.
"""

import torch
import torch.nn as nn
from monai.networks.nets import SwinUNETR

from swin3d_dnp.constants import DEFAULT_FINE_CHANNELS, DEFAULT_FINE_SHAPE


class FineNet(nn.Module):
    """Fine-resolution Swin3D network for patch processing.

    Processes high-resolution patches that have been fused with
    coarse context features. The input is the output of the
    CoarseContextFusion layer.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int = 1,
        feature_size: int = DEFAULT_FINE_CHANNELS,
        img_size: tuple[int, int, int] = DEFAULT_FINE_SHAPE,
        depths: tuple[int, ...] = (2, 2, 2, 2),
        num_heads: tuple[int, ...] = (3, 6, 12, 24),
        drop_rate: float = 0.0,
        attn_drop_rate: float = 0.0,
        use_checkpoint: bool = False,
        spatial_dims: int = 3,
    ):
        """Initialize FineNet.

        Args:
            in_channels: Number of input channels (from fusion layer output).
            out_channels: Number of output channels (task-specific).
            feature_size: Base feature channel count.
            img_size: Expected input spatial size (Df, Hf, Wf).
            depths: Number of Swin blocks at each stage.
            num_heads: Number of attention heads at each stage.
            drop_rate: Dropout rate.
            attn_drop_rate: Attention dropout rate.
            use_checkpoint: Use gradient checkpointing to save memory.
            spatial_dims: Number of spatial dimensions.
        """
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.feature_size = feature_size

        # Main SwinUNETR backbone
        self.swin_unetr = SwinUNETR(
            img_size=img_size,
            in_channels=in_channels,
            out_channels=out_channels,
            feature_size=feature_size,
            depths=depths,
            num_heads=num_heads,
            drop_rate=drop_rate,
            attn_drop_rate=attn_drop_rate,
            use_checkpoint=use_checkpoint,
            spatial_dims=spatial_dims,
            normalize=True,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through fine network.

        Args:
            x: Input tensor of shape (B, in_channels, Df, Hf, Wf)
               This is the output of the CoarseContextFusion layer.

        Returns:
            logits: (B, out_channels, Df, Hf, Wf) fine-resolution predictions.
        """
        return self.swin_unetr(x)


class FineNetLite(nn.Module):
    """Lightweight fine network for testing and small-scale experiments.

    Uses smaller feature sizes compared to full FineNet.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int = 1,
        feature_size: int = 24,
        img_size: tuple[int, int, int] = (32, 32, 32),
    ):
        """Initialize lightweight FineNet.

        Args:
            in_channels: Number of input channels.
            out_channels: Number of output channels.
            feature_size: Base feature channel count.
            img_size: Expected input spatial size.
        """
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.feature_size = feature_size

        self.swin_unetr = SwinUNETR(
            img_size=img_size,
            in_channels=in_channels,
            out_channels=out_channels,
            feature_size=feature_size,
            depths=(2, 2, 2, 2),
            num_heads=(3, 6, 12, 24),
            use_checkpoint=False,
            spatial_dims=3,
            normalize=True,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        return self.swin_unetr(x)


class SimpleFineNet(nn.Module):
    """Simple convolutional fine network for fast testing.

    Uses standard 3D convolutions instead of Swin Transformer.
    Much faster but less powerful than full SwinUNETR.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int = 1,
        base_channels: int = 32,
    ):
        """Initialize simple fine network.

        Args:
            in_channels: Number of input channels.
            out_channels: Number of output channels.
            base_channels: Base number of convolutional channels.
        """
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels

        self.encoder = nn.Sequential(
            nn.Conv3d(in_channels, base_channels, kernel_size=3, padding=1),
            nn.InstanceNorm3d(base_channels, affine=True),
            nn.GELU(),
            nn.Conv3d(base_channels, base_channels * 2, kernel_size=3, padding=1, stride=2),
            nn.InstanceNorm3d(base_channels * 2, affine=True),
            nn.GELU(),
            nn.Conv3d(base_channels * 2, base_channels * 4, kernel_size=3, padding=1, stride=2),
            nn.InstanceNorm3d(base_channels * 4, affine=True),
            nn.GELU(),
        )

        self.decoder = nn.Sequential(
            nn.ConvTranspose3d(base_channels * 4, base_channels * 2, kernel_size=2, stride=2),
            nn.InstanceNorm3d(base_channels * 2, affine=True),
            nn.GELU(),
            nn.ConvTranspose3d(base_channels * 2, base_channels, kernel_size=2, stride=2),
            nn.InstanceNorm3d(base_channels, affine=True),
            nn.GELU(),
            nn.Conv3d(base_channels, out_channels, kernel_size=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        z = self.encoder(x)
        return self.decoder(z)
