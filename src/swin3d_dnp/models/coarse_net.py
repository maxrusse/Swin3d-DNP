"""Coarse network wrapper for Swin3D encoder/decoder.

This module wraps MONAI's SwinUNETR for coarse-resolution processing,
providing both task logits and intermediate features for context fusion.
"""

import torch
import torch.nn as nn
from monai.networks.nets import SwinUNETR

from swin3d_dnp.constants import DEFAULT_COARSE_CHANNELS, DEFAULT_COARSE_SHAPE


class CoarseNet(nn.Module):
    """Coarse-resolution Swin3D network.

    Wraps MONAI's SwinUNETR to provide:
    - Multi-task logits (landmarks, organs, lesions)
    - Intermediate encoder features for context fusion with fine network

    The feature extraction happens at the decoder bottleneck, providing
    rich semantic features that can be sampled and fused with fine patches.
    """

    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 1,
        feature_size: int = DEFAULT_COARSE_CHANNELS,
        img_size: tuple[int, int, int] = DEFAULT_COARSE_SHAPE,
        depths: tuple[int, ...] = (2, 2, 2, 2),
        num_heads: tuple[int, ...] = (3, 6, 12, 24),
        drop_rate: float = 0.0,
        attn_drop_rate: float = 0.0,
        use_checkpoint: bool = False,
        spatial_dims: int = 3,
    ):
        """Initialize CoarseNet.

        Args:
            in_channels: Number of input channels.
            out_channels: Number of output channels (Nlm + Corg + 1 typically).
            feature_size: Base feature channel count (default 48).
            img_size: Expected input spatial size (D, H, W).
            depths: Number of Swin blocks at each stage.
            num_heads: Number of attention heads at each stage.
            drop_rate: Dropout rate.
            attn_drop_rate: Attention dropout rate.
            use_checkpoint: Use gradient checkpointing to save memory.
            spatial_dims: Number of spatial dimensions (always 3).
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

        # Feature extraction from encoder's deepest stage
        # SwinUNETR has feature sizes: feature_size, 2*feature_size, 4*feature_size, 8*feature_size
        # We extract from the bottleneck which has 8*feature_size channels
        self.feature_channels = 8 * feature_size

        # Feature projection to a manageable size for context sampling
        self.feature_proj = nn.Sequential(
            nn.Conv3d(self.feature_channels, feature_size, kernel_size=1, bias=False),
            nn.InstanceNorm3d(feature_size, affine=True),
            nn.GELU(),
        )

    def forward(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass through coarse network.

        Args:
            x: Input tensor of shape (B, C, D, H, W).

        Returns:
            Tuple of:
                - logits: (B, out_channels, D, H, W) task predictions
                - features: (B, feature_size, D', H', W') context features
                  where D', H', W' are spatially downsampled
        """
        # Get encoder hidden states
        hidden_states = self.swin_unetr.swinViT(x, self.swin_unetr.normalize)

        # hidden_states is a list of encoder outputs at each stage:
        # [stage0, stage1, stage2, stage3, stage4]
        # stage4 is the bottleneck with shape (B, 8*feature_size, D/32, H/32, W/32)

        # Extract bottleneck features
        enc_bottleneck = hidden_states[4]

        # Get decoder output (full resolution logits)
        # We need to run the decoder manually to get logits
        enc0 = hidden_states[0]  # (B, feature_size, D/2, H/2, W/2)
        enc1 = hidden_states[1]  # (B, 2*feature_size, D/4, H/4, W/4)
        enc2 = hidden_states[2]  # (B, 4*feature_size, D/8, H/8, W/8)
        enc3 = hidden_states[3]  # (B, 8*feature_size, D/16, H/16, W/16)

        # Run through decoder stages
        dec4 = self.swin_unetr.encoder10(enc_bottleneck)
        dec3 = self.swin_unetr.decoder5(dec4, enc3)
        dec2 = self.swin_unetr.decoder4(dec3, enc2)
        dec1 = self.swin_unetr.decoder3(dec2, enc1)
        dec0 = self.swin_unetr.decoder2(dec1, enc0)
        out = self.swin_unetr.decoder1(dec0)
        logits = self.swin_unetr.out(out)

        # Project bottleneck features for context fusion
        # Features are at 1/32 resolution with rich semantic information
        features = self.feature_proj(enc_bottleneck)

        return logits, features

    def get_feature_channels(self) -> int:
        """Return the number of channels in output features."""
        return self.feature_size


class CoarseNetLite(nn.Module):
    """Lightweight coarse network for testing and small-scale experiments.

    Uses smaller feature sizes and depths compared to full CoarseNet.
    """

    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 1,
        feature_size: int = 24,
        img_size: tuple[int, int, int] = (64, 64, 64),
    ):
        """Initialize lightweight CoarseNet.

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

        self.feature_channels = 8 * feature_size
        self.feature_proj = nn.Sequential(
            nn.Conv3d(self.feature_channels, feature_size, kernel_size=1, bias=False),
            nn.InstanceNorm3d(feature_size, affine=True),
            nn.GELU(),
        )

    def forward(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass."""
        hidden_states = self.swin_unetr.swinViT(x, self.swin_unetr.normalize)

        enc_bottleneck = hidden_states[4]
        enc0 = hidden_states[0]
        enc1 = hidden_states[1]
        enc2 = hidden_states[2]
        enc3 = hidden_states[3]

        dec4 = self.swin_unetr.encoder10(enc_bottleneck)
        dec3 = self.swin_unetr.decoder5(dec4, enc3)
        dec2 = self.swin_unetr.decoder4(dec3, enc2)
        dec1 = self.swin_unetr.decoder3(dec2, enc1)
        dec0 = self.swin_unetr.decoder2(dec1, enc0)
        out = self.swin_unetr.decoder1(dec0)
        logits = self.swin_unetr.out(out)

        features = self.feature_proj(enc_bottleneck)

        return logits, features

    def get_feature_channels(self) -> int:
        """Return the number of channels in output features."""
        return self.feature_size
