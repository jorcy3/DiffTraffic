from __future__ import annotations

import torch
from torch import nn

from .masking import SpatioTemporalMaskGenerator


class FactorizedSelfAttentionBlock(nn.Module):
    """
    Lightweight factorized temporal-spatial attention block for enhancement features.
    """

    def __init__(
        self,
        model_dim: int,
        num_heads: int,
        feed_forward_dim: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.temporal_attn = nn.MultiheadAttention(
            embed_dim=model_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.spatial_attn = nn.MultiheadAttention(
            embed_dim=model_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.temporal_ln1 = nn.LayerNorm(model_dim)
        self.temporal_ln2 = nn.LayerNorm(model_dim)
        self.spatial_ln1 = nn.LayerNorm(model_dim)
        self.spatial_ln2 = nn.LayerNorm(model_dim)
        self.temporal_ffn = nn.Sequential(
            nn.Linear(model_dim, feed_forward_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(feed_forward_dim, model_dim),
        )
        self.spatial_ffn = nn.Sequential(
            nn.Linear(model_dim, feed_forward_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(feed_forward_dim, model_dim),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Shape:
            x: [B, I, N, D]
            return: [B, I, N, D]
        """

        batch_size, input_len, num_nodes, model_dim = x.shape

        temporal_tokens = x.permute(0, 2, 1, 3).reshape(batch_size * num_nodes, input_len, model_dim)
        temporal_out, _ = self.temporal_attn(temporal_tokens, temporal_tokens, temporal_tokens, need_weights=False)
        temporal_tokens = self.temporal_ln1(temporal_tokens + self.dropout(temporal_out))
        temporal_tokens = self.temporal_ln2(temporal_tokens + self.dropout(self.temporal_ffn(temporal_tokens)))
        x = temporal_tokens.reshape(batch_size, num_nodes, input_len, model_dim).permute(0, 2, 1, 3)

        spatial_tokens = x.reshape(batch_size * input_len, num_nodes, model_dim)
        spatial_out, _ = self.spatial_attn(spatial_tokens, spatial_tokens, spatial_tokens, need_weights=False)
        spatial_tokens = self.spatial_ln1(spatial_tokens + self.dropout(spatial_out))
        spatial_tokens = self.spatial_ln2(spatial_tokens + self.dropout(self.spatial_ffn(spatial_tokens)))
        return spatial_tokens.reshape(batch_size, input_len, num_nodes, model_dim)


class STMAEStyleEnhancer(nn.Module):
    """
    Minimal STMAE-style enhancement encoder shared by masked reconstruction and forecasting.
    """

    def __init__(
        self,
        *,
        num_nodes: int,
        input_len: int,
        steps_per_day: int,
        num_day_in_week: int,
        value_embedding_dim: int,
        tod_embedding_dim: int,
        dow_embedding_dim: int,
        node_embedding_dim: int,
        model_dim: int,
        num_heads: int,
        num_layers: int,
        feed_forward_dim: int,
        dropout: float,
        spatial_mask_ratio: float,
        temporal_mask_ratio: float,
        temporal_patch_size: int,
        enhancement_scale: float,
        mask_value: float,
    ) -> None:
        super().__init__()
        self.num_nodes = num_nodes
        self.input_len = input_len
        self.steps_per_day = steps_per_day
        self.num_day_in_week = num_day_in_week
        self.tod_embedding_dim = tod_embedding_dim
        self.dow_embedding_dim = dow_embedding_dim
        self.node_embedding_dim = node_embedding_dim
        self.enhancement_scale = float(enhancement_scale)
        self.mask_value = float(mask_value)

        feature_dim = value_embedding_dim + tod_embedding_dim + dow_embedding_dim + node_embedding_dim
        if feature_dim != model_dim:
            raise ValueError(f"model_dim={model_dim} must equal concatenated feature dim={feature_dim}.")
        if model_dim % num_heads != 0:
            raise ValueError(f"model_dim ({model_dim}) must be divisible by num_heads ({num_heads}).")

        self.value_proj = nn.Linear(1, value_embedding_dim)
        self.tod_embedding = nn.Embedding(steps_per_day, tod_embedding_dim) if tod_embedding_dim > 0 else None
        self.dow_embedding = nn.Embedding(num_day_in_week, dow_embedding_dim) if dow_embedding_dim > 0 else None
        self.node_embedding = (
            nn.Parameter(torch.empty(num_nodes, node_embedding_dim)) if node_embedding_dim > 0 else None
        )
        if self.node_embedding is not None:
            nn.init.xavier_uniform_(self.node_embedding)

        self.mask_generator = SpatioTemporalMaskGenerator(
            spatial_mask_ratio=spatial_mask_ratio,
            temporal_mask_ratio=temporal_mask_ratio,
            temporal_patch_size=temporal_patch_size,
        )
        self.encoder_blocks = nn.ModuleList(
            [
                FactorizedSelfAttentionBlock(
                    model_dim=model_dim,
                    num_heads=num_heads,
                    feed_forward_dim=feed_forward_dim,
                    dropout=dropout,
                )
                for _ in range(num_layers)
            ]
        )
        self.reconstruction_head = nn.Sequential(
            nn.LayerNorm(model_dim),
            nn.Linear(model_dim, model_dim),
            nn.GELU(),
            nn.Linear(model_dim, 1),
        )
        self.enhancement_head = nn.Sequential(
            nn.LayerNorm(model_dim),
            nn.Linear(model_dim, model_dim),
            nn.GELU(),
            nn.Linear(model_dim, 1),
        )

    def _recover_time_indices(self, inputs_timestamps: torch.Tensor | None) -> tuple[torch.Tensor, torch.Tensor]:
        if inputs_timestamps is None:
            zeros = torch.zeros(1, self.input_len, dtype=torch.long)
            return zeros, zeros

        tod_index = torch.floor(inputs_timestamps[..., 0] * self.steps_per_day).long()
        tod_index = tod_index.clamp(min=0, max=self.steps_per_day - 1)
        dow_index = torch.floor(inputs_timestamps[..., 1] * self.num_day_in_week + 1e-6).long()
        dow_index = dow_index.clamp(min=0, max=self.num_day_in_week - 1)
        return tod_index, dow_index

    def _build_features(
        self,
        inputs: torch.Tensor,
        inputs_timestamps: torch.Tensor | None,
    ) -> torch.Tensor:
        batch_size, input_len, num_nodes = inputs.shape
        features = [self.value_proj(inputs.unsqueeze(-1))]

        tod_index, dow_index = self._recover_time_indices(inputs_timestamps)
        if self.tod_embedding is not None:
            tod_index = tod_index.to(inputs.device)
            if tod_index.shape[0] == 1 and batch_size > 1:
                tod_index = tod_index.expand(batch_size, -1)
            features.append(self.tod_embedding(tod_index).unsqueeze(2).expand(-1, -1, num_nodes, -1))
        if self.dow_embedding is not None:
            dow_index = dow_index.to(inputs.device)
            if dow_index.shape[0] == 1 and batch_size > 1:
                dow_index = dow_index.expand(batch_size, -1)
            features.append(self.dow_embedding(dow_index).unsqueeze(2).expand(-1, -1, num_nodes, -1))
        if self.node_embedding is not None:
            node_embedding = self.node_embedding.unsqueeze(0).unsqueeze(0).expand(batch_size, input_len, -1, -1)
            features.append(node_embedding)
        return torch.cat(features, dim=-1)

    def _encode(
        self,
        inputs: torch.Tensor,
        inputs_timestamps: torch.Tensor | None,
    ) -> torch.Tensor:
        hidden = self._build_features(inputs, inputs_timestamps)
        for block in self.encoder_blocks:
            hidden = block(hidden)
        return hidden

    def forward(
        self,
        inputs: torch.Tensor,
        inputs_timestamps: torch.Tensor | None = None,
        input_valid_mask: torch.Tensor | None = None,
        force_mask: bool = False,
    ) -> dict[str, torch.Tensor]:
        """
        Shape:
            inputs: [B, I, N]
            inputs_timestamps: [B, I, 2] or None
            input_valid_mask: [B, I, N] or None
            force_mask: bool
            return:
                enhanced_inputs: [B, I, N]
                masked_reconstruction: [B, I, N]
                masked_positions: [B, I, N]
                enhancement_delta: [B, I, N]
        """

        masked_positions = (
            self.mask_generator(inputs, valid_mask=input_valid_mask)
            if self.training or force_mask
            else torch.zeros_like(inputs, dtype=torch.bool)
        )
        masked_inputs = torch.where(masked_positions, torch.full_like(inputs, self.mask_value), inputs)

        masked_hidden = self._encode(masked_inputs, inputs_timestamps)
        full_hidden = self._encode(inputs, inputs_timestamps)

        masked_reconstruction = self.reconstruction_head(masked_hidden).squeeze(-1)
        enhancement_delta = torch.tanh(self.enhancement_head(full_hidden).squeeze(-1))
        enhanced_inputs = inputs + self.enhancement_scale * enhancement_delta

        return {
            "enhanced_inputs": enhanced_inputs,
            "masked_reconstruction": masked_reconstruction,
            "masked_positions": masked_positions,
            "enhancement_delta": enhancement_delta,
        }
