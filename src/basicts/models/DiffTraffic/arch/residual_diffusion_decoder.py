from typing import Optional

import torch
from torch import nn

from ..config import DiffTrafficV0Config


class ResidualDiffusionDecoder(nn.Module):
    """
    Lightweight residual decoder shell.

    v0 responsibility:
    - accept a fused condition
    - return residual_prediction with the same shape as base_prediction
    - keep diffusion-specific details out of the backbone
    """

    def __init__(self, config: DiffTrafficV0Config):
        super().__init__()
        self.config = config
        self.num_features = config.num_features
        hidden_size = config.residual_hidden_size
        dropout = config.residual_dropout

        self.residual_head = nn.Sequential(
            nn.Linear(self.num_features, hidden_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, self.num_features),
        )

        last_layer = self.residual_head[-1]
        nn.init.zeros_(last_layer.weight)
        nn.init.zeros_(last_layer.bias)

    def forward(
        self,
        condition: dict,
        step: Optional[int] = None,
        epoch: Optional[int] = None,
        train: Optional[bool] = None,
    ) -> dict:
        """
        Args:
            condition: dict returned by ConditionFusion
            step: optional global step
            epoch: optional epoch index
            train: optional training flag

        Returns:
            dict with:
                residual_prediction: [batch_size, output_len, num_features]
                aux_info: auxiliary metadata.
        """

        base_prediction = condition["base_prediction"]
        residual_prediction = self.residual_head(base_prediction)
        return {
            "residual_prediction": residual_prediction,
            "aux_info": {
                "step": step,
                "epoch": epoch,
                "train": train,
                "condition_keys": list(condition.keys()),
            },
        }

