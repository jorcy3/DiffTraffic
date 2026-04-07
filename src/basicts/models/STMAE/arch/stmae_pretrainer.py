from __future__ import annotations

import torch
from torch import nn

from ..config.stmae_config import STMAEConfig
from .stmae_encoder import STMAEStyleEnhancer


class STMAEPretrainer(nn.Module):
    """
    STMAE-style masked pretrainer for representation enhancement.
    """

    def __init__(self, config: STMAEConfig) -> None:
        super().__init__()
        self.config = config
        self.enhancer = STMAEStyleEnhancer(
            num_nodes=config.num_features,
            input_len=config.input_len,
            steps_per_day=config.steps_per_day,
            num_day_in_week=config.num_day_in_week,
            value_embedding_dim=config.enhancement_value_embedding_dim,
            tod_embedding_dim=config.enhancement_tod_embedding_dim,
            dow_embedding_dim=config.enhancement_dow_embedding_dim,
            node_embedding_dim=config.enhancement_node_embedding_dim,
            model_dim=config.enhancement_model_dim,
            num_heads=config.enhancement_num_heads,
            num_layers=config.enhancement_num_layers,
            feed_forward_dim=config.enhancement_feed_forward_dim,
            dropout=config.enhancement_dropout,
            spatial_mask_ratio=config.spatial_mask_ratio,
            temporal_mask_ratio=config.temporal_mask_ratio,
            temporal_patch_size=config.temporal_patch_size,
            enhancement_scale=config.enhancement_scale,
            mask_value=config.mask_value,
        )
        self.loss_weights = {
            "prediction": 0.0,
            "reconstruction": float(config.loss_weight_reconstruction),
        }

    def forward(
        self,
        inputs: torch.Tensor,
        inputs_timestamps: torch.Tensor | None = None,
        enhancement_valid_mask: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor | dict]:
        """
        Shape:
            inputs: [B, I, N]
            inputs_timestamps: [B, I, 2] or None
            enhancement_valid_mask: [B, I, N] or None
            return:
                prediction: [B, I, N]
                masked_positions: [B, I, N]
        """

        outputs = self.enhancer(
            inputs=inputs,
            inputs_timestamps=inputs_timestamps,
            input_valid_mask=enhancement_valid_mask,
            force_mask=True,
        )
        return {
            "prediction": outputs["masked_reconstruction"],
            "masked_reconstruction": outputs["masked_reconstruction"],
            "masked_positions": outputs["masked_positions"],
            "enhancement_delta": outputs["enhancement_delta"],
            "enhanced_inputs": outputs["enhanced_inputs"],
            "aux_info": {
                "loss_weights": self.loss_weights,
            },
        }
