"""Context fusion layer for combining image and coarse features.

This module implements the fusion of:
- Fine-resolution image patches
- Coarse context features (sampled at same physical FOV)
- Optional coarse probability maps
"""

import torch
import torch.nn as nn


class CoarseContextFusion(nn.Module):
    """Fuse fine image with coarse context features.

    Combines the fine-resolution input image with coarse context features
    that have been sampled to cover the same physical field of view.
    Optionally incorporates coarse probability predictions for additional
    guidance.

    The output is suitable for input to the fine network.
    """

    def __init__(
        self,
        in_channels_image: int = 1,
        in_channels_context: int = 64,
        in_channels_probs: int | None = None,
        out_channels: int = 64,
        norm: str = "instance",
    ):
        """Initialize fusion layer.

        Args:
            in_channels_image: Number of image channels (typically 1).
            in_channels_context: Number of context feature channels.
            in_channels_probs: Number of probability channels (optional).
                If None, probabilities are not used.
            out_channels: Number of output channels.
            norm: Normalization type ("instance", "batch", or "layer").
        """
        super().__init__()

        self.in_channels_image = in_channels_image
        self.in_channels_context = in_channels_context
        self.in_channels_probs = in_channels_probs
        self.out_channels = out_channels

        # Calculate total input channels
        total_in = in_channels_image + in_channels_context
        self.use_probs = in_channels_probs is not None
        if self.use_probs:
            total_in += in_channels_probs

        # Projection layer
        self.proj = nn.Conv3d(total_in, out_channels, kernel_size=1, bias=False)

        # Normalization
        if norm == "instance":
            self.norm = nn.InstanceNorm3d(out_channels, affine=True)
        elif norm == "batch":
            self.norm = nn.BatchNorm3d(out_channels)
        elif norm == "layer":
            self.norm = nn.GroupNorm(1, out_channels)
        else:
            raise ValueError(f"Unknown norm type: {norm}. Use 'instance', 'batch', or 'layer'.")

        # Activation
        self.act = nn.GELU()

    def forward(
        self,
        image_fine: torch.Tensor,
        context_feat: torch.Tensor,
        coarse_logits_fine: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Fuse image, context, and optional probabilities.

        Args:
            image_fine: (B, C_img, D, H, W) fine-resolution image patch.
            context_feat: (B, C_ctx, D, H, W) coarse context features
                sampled at same physical FOV as image_fine.
            coarse_logits_fine: (B, C_probs, D, H, W) coarse logits
                sampled at same physical FOV (optional).

        Returns:
            (B, out_channels, D, H, W) fused features for fine network.
        """
        parts = [image_fine, context_feat]

        if self.use_probs and coarse_logits_fine is not None:
            # Convert logits to probabilities
            if coarse_logits_fine.shape[1] > 1:
                probs = torch.softmax(coarse_logits_fine, dim=1)
            else:
                probs = torch.sigmoid(coarse_logits_fine)
            parts.append(probs)

        # Concatenate along channel dimension
        x = torch.cat(parts, dim=1)

        # Project, normalize, activate
        x = self.proj(x)
        x = self.norm(x)
        x = self.act(x)

        return x


class AdaptiveCoarseContextFusion(nn.Module):
    """Adaptive fusion with learned attention weights.

    More sophisticated fusion that uses channel attention to
    adaptively weight the importance of image vs context features.
    """

    def __init__(
        self,
        in_channels_image: int = 1,
        in_channels_context: int = 64,
        in_channels_probs: int | None = None,
        out_channels: int = 64,
        reduction: int = 4,
        norm: str = "instance",
    ):
        """Initialize adaptive fusion layer.

        Args:
            in_channels_image: Number of image channels.
            in_channels_context: Number of context feature channels.
            in_channels_probs: Number of probability channels (optional).
            out_channels: Number of output channels.
            reduction: Channel reduction ratio for attention.
            norm: Normalization type.
        """
        super().__init__()

        self.in_channels_image = in_channels_image
        self.in_channels_context = in_channels_context
        self.in_channels_probs = in_channels_probs
        self.out_channels = out_channels

        # Calculate total input channels
        total_in = in_channels_image + in_channels_context
        self.use_probs = in_channels_probs is not None
        if self.use_probs:
            total_in += in_channels_probs

        # Initial projection
        self.proj = nn.Conv3d(total_in, out_channels, kernel_size=1, bias=False)

        # Channel attention
        hidden_dim = max(out_channels // reduction, 8)
        self.attention = nn.Sequential(
            nn.AdaptiveAvgPool3d(1),
            nn.Flatten(1),
            nn.Linear(out_channels, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, out_channels),
            nn.Sigmoid(),
        )

        # Normalization
        if norm == "instance":
            self.norm = nn.InstanceNorm3d(out_channels, affine=True)
        elif norm == "batch":
            self.norm = nn.BatchNorm3d(out_channels)
        elif norm == "layer":
            self.norm = nn.GroupNorm(1, out_channels)
        else:
            raise ValueError(f"Unknown norm type: {norm}")

        self.act = nn.GELU()

    def forward(
        self,
        image_fine: torch.Tensor,
        context_feat: torch.Tensor,
        coarse_logits_fine: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Fuse with adaptive attention weighting.

        Args:
            image_fine: (B, C_img, D, H, W) fine image patch.
            context_feat: (B, C_ctx, D, H, W) coarse context features.
            coarse_logits_fine: (B, C_probs, D, H, W) coarse logits (optional).

        Returns:
            (B, out_channels, D, H, W) adaptively fused features.
        """
        parts = [image_fine, context_feat]

        if self.use_probs and coarse_logits_fine is not None:
            if coarse_logits_fine.shape[1] > 1:
                probs = torch.softmax(coarse_logits_fine, dim=1)
            else:
                probs = torch.sigmoid(coarse_logits_fine)
            parts.append(probs)

        x = torch.cat(parts, dim=1)
        x = self.proj(x)

        # Channel attention
        B = x.shape[0]
        attn = self.attention(x).view(B, -1, 1, 1, 1)
        x = x * attn

        x = self.norm(x)
        x = self.act(x)

        return x


class SimpleFusion(nn.Module):
    """Simple concatenation-based fusion for testing.

    Just concatenates inputs without learned projection.
    Useful for debugging and quick experiments.
    """

    def __init__(
        self,
        in_channels_image: int = 1,
        in_channels_context: int = 64,
        in_channels_probs: int | None = None,
    ):
        """Initialize simple fusion.

        Args:
            in_channels_image: Number of image channels.
            in_channels_context: Number of context feature channels.
            in_channels_probs: Number of probability channels (optional).
        """
        super().__init__()

        self.in_channels_image = in_channels_image
        self.in_channels_context = in_channels_context
        self.in_channels_probs = in_channels_probs

        total_in = in_channels_image + in_channels_context
        self.use_probs = in_channels_probs is not None
        if self.use_probs:
            total_in += in_channels_probs

        self.out_channels = total_in

    def forward(
        self,
        image_fine: torch.Tensor,
        context_feat: torch.Tensor,
        coarse_logits_fine: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Simple concatenation fusion."""
        parts = [image_fine, context_feat]

        if self.use_probs and coarse_logits_fine is not None:
            if coarse_logits_fine.shape[1] > 1:
                probs = torch.softmax(coarse_logits_fine, dim=1)
            else:
                probs = torch.sigmoid(coarse_logits_fine)
            parts.append(probs)

        return torch.cat(parts, dim=1)
