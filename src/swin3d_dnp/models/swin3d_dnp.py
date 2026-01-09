"""Main Swin3D-DNP model combining coarse and fine networks.

This module implements the complete hierarchical model with:
- Coarse network for global context and initial predictions
- Fine network for high-resolution patch refinement
- Differentiable context sampling and fusion
- Phase-controlled gradient flow for training
"""

import torch
import torch.nn as nn

from swin3d_dnp.geometry.sampling import (
    DifferentiableContextSampler,
    extent_vox_in_src_from_spacings,
)
from swin3d_dnp.models.coarse_net import CoarseNet, CoarseNetLite
from swin3d_dnp.models.fine_net import FineNet, FineNetLite, SimpleFineNet
from swin3d_dnp.models.fusion import CoarseContextFusion, SimpleFusion


class Swin3DDNP(nn.Module):
    """Complete Swin3D-DNP hierarchical model.

    Combines coarse and fine networks with differentiable context sampling.
    Supports phase-controlled training with optional gradient detachment
    for memory efficiency during early training phases.

    Architecture:
        1. Coarse network produces global features and logits
        2. Context sampler extracts features at fine patch FOV
        3. Fusion layer combines fine image with context
        4. Fine network refines predictions at high resolution
    """

    def __init__(
        self,
        coarse_net: nn.Module,
        fine_net: nn.Module,
        fusion: nn.Module,
        context_sampler: DifferentiableContextSampler | None = None,
        detach_coarse_context: bool = False,
        sample_logits_for_fusion: bool = False,
    ):
        """Initialize Swin3D-DNP model.

        Args:
            coarse_net: Coarse resolution network (returns logits, features).
            fine_net: Fine resolution network.
            fusion: Fusion layer combining image and context.
            context_sampler: Differentiable context sampler (created if None).
            detach_coarse_context: Whether to detach coarse gradients.
            sample_logits_for_fusion: Whether to sample and use coarse logits
                in fusion (requires fusion layer to support probs).
        """
        super().__init__()

        self.coarse_net = coarse_net
        self.fine_net = fine_net
        self.fusion = fusion
        self.context_sampler = context_sampler or DifferentiableContextSampler()
        self.detach_coarse_context = detach_coarse_context
        self.sample_logits_for_fusion = sample_logits_for_fusion

        # Training phase (1=warmup, 2=transition, 3=final)
        self._phase = 1

    def set_phase(self, phase: int) -> None:
        """Set training phase for gradient control.

        Phase 1 (warmup): Detach coarse context, focus on coarse learning
        Phase 2 (transition): Enable gradients, begin end-to-end training
        Phase 3 (final): Full end-to-end with fine-tuning

        Args:
            phase: Training phase (1, 2, or 3).
        """
        if phase not in (1, 2, 3):
            raise ValueError(f"Phase must be 1, 2, or 3, got {phase}")

        self._phase = phase

        # Detach coarse context in phase 1 for memory efficiency
        self.detach_coarse_context = (phase == 1)

    def get_phase(self) -> int:
        """Return current training phase."""
        return self._phase

    def forward(
        self,
        image_coarse: torch.Tensor,
        image_fine: torch.Tensor,
        centers_coarse_norm_dhw: torch.Tensor,
        fine_shape: tuple[int, int, int],
        spacing_fine_dhw_mm: torch.Tensor,
        spacing_coarse_dhw_mm: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass through complete model.

        Args:
            image_coarse: (B, 1, Dc, Hc, Wc) coarse-resolution image.
            image_fine: (B, 1, Df, Hf, Wf) fine-resolution patch.
            centers_coarse_norm_dhw: (B, 3) normalized patch centers
                in coarse volume space (d, h, w order).
            fine_shape: (Df, Hf, Wf) fine patch shape.
            spacing_fine_dhw_mm: (B, 3) or (3,) fine voxel spacing in mm.
            spacing_coarse_dhw_mm: (B, 3) or (3,) coarse voxel spacing in mm.

        Returns:
            Tuple of:
                - coarse_logits: (B, C_out, Dc, Hc, Wc) coarse predictions
                - fine_logits: (B, C_out, Df, Hf, Wf) fine predictions
        """
        # 1. Coarse forward
        coarse_logits, coarse_feat = self.coarse_net(image_coarse)

        # 2. Compute extent in coarse voxels for context sampling
        extent_vox = extent_vox_in_src_from_spacings(
            fine_shape,
            spacing_fine_dhw_mm,
            spacing_coarse_dhw_mm,
        )

        # 3. Optionally detach for memory efficiency
        if self.detach_coarse_context:
            cf = coarse_feat.detach()
            cl = coarse_logits.detach()
        else:
            cf = coarse_feat
            cl = coarse_logits

        # 4. Sample context at fine patch FOV
        context = self.context_sampler(
            cf, centers_coarse_norm_dhw, fine_shape, extent_vox
        )

        # 5. Optionally sample coarse logits for fusion
        coarse_logits_fine = None
        if self.sample_logits_for_fusion:
            coarse_logits_fine = self.context_sampler(
                cl, centers_coarse_norm_dhw, fine_shape, extent_vox
            )

        # 6. Fuse image with context
        fused = self.fusion(image_fine, context, coarse_logits_fine)

        # 7. Fine network forward
        fine_logits = self.fine_net(fused)

        return coarse_logits, fine_logits

    def forward_coarse_only(
        self, image_coarse: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass through coarse network only.

        Useful for inference when generating proposals.

        Args:
            image_coarse: (B, 1, Dc, Hc, Wc) coarse image.

        Returns:
            Tuple of (coarse_logits, coarse_features).
        """
        return self.coarse_net(image_coarse)

    def forward_fine_with_context(
        self,
        image_fine: torch.Tensor,
        coarse_feat: torch.Tensor,
        coarse_logits: torch.Tensor,
        centers_coarse_norm_dhw: torch.Tensor,
        fine_shape: tuple[int, int, int],
        spacing_fine_dhw_mm: torch.Tensor,
        spacing_coarse_dhw_mm: torch.Tensor,
    ) -> torch.Tensor:
        """Forward through fine network with pre-computed coarse features.

        Useful for inference when processing multiple patches from
        the same coarse volume.

        Args:
            image_fine: (B, 1, Df, Hf, Wf) fine patch.
            coarse_feat: (1, C, Dc', Hc', Wc') pre-computed coarse features.
            coarse_logits: (1, C, Dc, Hc, Wc) pre-computed coarse logits.
            centers_coarse_norm_dhw: (B, 3) normalized centers.
            fine_shape: (Df, Hf, Wf) patch shape.
            spacing_fine_dhw_mm: Fine spacing in mm.
            spacing_coarse_dhw_mm: Coarse spacing in mm.

        Returns:
            fine_logits: (B, C_out, Df, Hf, Wf) fine predictions.
        """
        extent_vox = extent_vox_in_src_from_spacings(
            fine_shape,
            spacing_fine_dhw_mm,
            spacing_coarse_dhw_mm,
        )

        # Expand coarse features if batch size differs
        B = image_fine.shape[0]
        if coarse_feat.shape[0] == 1 and B > 1:
            cf = coarse_feat.expand(B, -1, -1, -1, -1)
            cl = coarse_logits.expand(B, -1, -1, -1, -1)
        else:
            cf = coarse_feat
            cl = coarse_logits

        context = self.context_sampler(
            cf, centers_coarse_norm_dhw, fine_shape, extent_vox
        )

        coarse_logits_fine = None
        if self.sample_logits_for_fusion:
            coarse_logits_fine = self.context_sampler(
                cl, centers_coarse_norm_dhw, fine_shape, extent_vox
            )

        fused = self.fusion(image_fine, context, coarse_logits_fine)
        return self.fine_net(fused)


def build_swin3d_dnp(
    in_channels: int = 1,
    out_channels: int = 1,
    coarse_feature_size: int = 48,
    fine_feature_size: int = 48,
    coarse_img_size: tuple[int, int, int] = (128, 128, 128),
    fine_img_size: tuple[int, int, int] = (96, 96, 96),
    use_checkpoint: bool = False,
    use_probs_fusion: bool = False,
    norm: str = "instance",
) -> Swin3DDNP:
    """Build a complete Swin3D-DNP model with default configuration.

    Args:
        in_channels: Number of input image channels.
        out_channels: Number of output channels (task-specific).
        coarse_feature_size: Base feature size for coarse network.
        fine_feature_size: Base feature size for fine network.
        coarse_img_size: Coarse volume shape (D, H, W).
        fine_img_size: Fine patch shape (D, H, W).
        use_checkpoint: Use gradient checkpointing.
        use_probs_fusion: Include coarse probabilities in fusion.
        norm: Normalization type for fusion layer.

    Returns:
        Configured Swin3DDNP model.
    """
    coarse_net = CoarseNet(
        in_channels=in_channels,
        out_channels=out_channels,
        feature_size=coarse_feature_size,
        img_size=coarse_img_size,
        use_checkpoint=use_checkpoint,
    )

    # Fusion output channels = fine network input channels
    fusion_out_channels = fine_feature_size

    fusion = CoarseContextFusion(
        in_channels_image=in_channels,
        in_channels_context=coarse_net.get_feature_channels(),
        in_channels_probs=out_channels if use_probs_fusion else None,
        out_channels=fusion_out_channels,
        norm=norm,
    )

    fine_net = FineNet(
        in_channels=fusion_out_channels,
        out_channels=out_channels,
        feature_size=fine_feature_size,
        img_size=fine_img_size,
        use_checkpoint=use_checkpoint,
    )

    return Swin3DDNP(
        coarse_net=coarse_net,
        fine_net=fine_net,
        fusion=fusion,
        sample_logits_for_fusion=use_probs_fusion,
    )


def build_swin3d_dnp_lite(
    in_channels: int = 1,
    out_channels: int = 1,
    feature_size: int = 24,
    coarse_img_size: tuple[int, int, int] = (64, 64, 64),
    fine_img_size: tuple[int, int, int] = (32, 32, 32),
) -> Swin3DDNP:
    """Build a lightweight Swin3D-DNP for testing.

    Args:
        in_channels: Number of input channels.
        out_channels: Number of output channels.
        feature_size: Base feature size.
        coarse_img_size: Coarse volume shape.
        fine_img_size: Fine patch shape.

    Returns:
        Lightweight Swin3DDNP model.
    """
    coarse_net = CoarseNetLite(
        in_channels=in_channels,
        out_channels=out_channels,
        feature_size=feature_size,
        img_size=coarse_img_size,
    )

    fusion = CoarseContextFusion(
        in_channels_image=in_channels,
        in_channels_context=coarse_net.get_feature_channels(),
        out_channels=feature_size,
        norm="instance",
    )

    fine_net = FineNetLite(
        in_channels=feature_size,
        out_channels=out_channels,
        feature_size=feature_size,
        img_size=fine_img_size,
    )

    return Swin3DDNP(
        coarse_net=coarse_net,
        fine_net=fine_net,
        fusion=fusion,
    )


def build_simple_swin3d_dnp(
    in_channels: int = 1,
    out_channels: int = 1,
    context_channels: int = 32,
) -> Swin3DDNP:
    """Build a simple Swin3D-DNP for fast testing.

    Uses simple convolutional networks instead of Swin Transformers.

    Args:
        in_channels: Number of input channels.
        out_channels: Number of output channels.
        context_channels: Number of context feature channels.

    Returns:
        Simple test model.
    """
    # Simple mock coarse network
    class SimpleCoarseNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.encoder = nn.Sequential(
                nn.Conv3d(in_channels, context_channels, 3, padding=1),
                nn.InstanceNorm3d(context_channels),
                nn.GELU(),
                nn.Conv3d(context_channels, context_channels, 3, padding=1, stride=2),
                nn.InstanceNorm3d(context_channels),
                nn.GELU(),
            )
            self.decoder = nn.Sequential(
                nn.ConvTranspose3d(context_channels, context_channels, 2, stride=2),
                nn.InstanceNorm3d(context_channels),
                nn.GELU(),
                nn.Conv3d(context_channels, out_channels, 1),
            )
            self.feat_proj = nn.Conv3d(context_channels, context_channels, 1)

        def forward(self, x):
            feat = self.encoder(x)
            logits = self.decoder(feat)
            return logits, self.feat_proj(feat)

        def get_feature_channels(self):
            return context_channels

    coarse_net = SimpleCoarseNet()

    fusion = SimpleFusion(
        in_channels_image=in_channels,
        in_channels_context=context_channels,
    )

    fine_net = SimpleFineNet(
        in_channels=fusion.out_channels,
        out_channels=out_channels,
        base_channels=context_channels,
    )

    return Swin3DDNP(
        coarse_net=coarse_net,
        fine_net=fine_net,
        fusion=fusion,
    )
