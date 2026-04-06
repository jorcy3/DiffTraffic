from __future__ import annotations

import math

import torch
from torch import nn


class SpatioTemporalMaskGenerator(nn.Module):
    """
    Generate STMAE-style masks with a spatial node mask and a temporal patch mask.

    This minimal migration keeps the dual-mask structure but uses uniform spatial
    masking instead of the paper's biased random walk policy.
    """

    def __init__(
        self,
        spatial_mask_ratio: float = 0.2,
        temporal_mask_ratio: float = 0.2,
        temporal_patch_size: int = 3,
    ) -> None:
        super().__init__()
        self.spatial_mask_ratio = float(spatial_mask_ratio)
        self.temporal_mask_ratio = float(temporal_mask_ratio)
        self.temporal_patch_size = int(temporal_patch_size)
        if self.temporal_patch_size <= 0:
            raise ValueError("temporal_patch_size must be a positive integer.")

    def forward(
        self,
        inputs: torch.Tensor,
        valid_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Shape:
            inputs: [B, I, N]
            valid_mask: [B, I, N] or None
            return: [B, I, N] bool
        """

        if inputs.ndim != 3:
            raise ValueError(f"Expected `inputs` with shape [B, I, N], got {tuple(inputs.shape)}.")

        batch_size, input_len, num_nodes = inputs.shape
        device = inputs.device
        mask = torch.zeros(batch_size, input_len, num_nodes, dtype=torch.bool, device=device)

        if self.spatial_mask_ratio > 0.0:
            spatial_mask = torch.rand(batch_size, num_nodes, device=device) < self.spatial_mask_ratio
            mask = mask | spatial_mask.unsqueeze(1).expand(-1, input_len, -1)

        if self.temporal_mask_ratio > 0.0:
            num_patches = math.ceil(input_len / self.temporal_patch_size)
            patch_mask = torch.rand(batch_size, num_patches, device=device) < self.temporal_mask_ratio
            temporal_mask = patch_mask.repeat_interleave(self.temporal_patch_size, dim=1)[:, :input_len]
            mask = mask | temporal_mask.unsqueeze(-1).expand(-1, -1, num_nodes)

        if valid_mask is not None:
            mask = mask & valid_mask.to(torch.bool)

        return mask
