from typing import Optional

import torch
from torch import nn

from .dynamic_condition_encoder import DynamicConditionEncoder
from .horizon_condition_encoder import HorizonConditionEncoder
from .residual_prior_encoder import ResidualPriorEncoder


class SelectiveConditionEncoderStack(nn.Module):
    """
    Condition encoders for DiffTraffic-v1.1 selective refinement.
    """

    def __init__(self, config):
        super().__init__()
        hidden_size = getattr(config, "condition_hidden_size", None) or config.residual_hidden_size
        self.horizon_encoder = HorizonConditionEncoder(
            num_features=config.num_features,
            hidden_size=hidden_size,
            output_len=config.output_len,
            steps_per_day=getattr(config, "steps_per_day", 288),
            num_day_in_week=getattr(config, "num_day_in_week", 7),
        )
        self.dynamic_encoder = DynamicConditionEncoder(config.num_features, hidden_size)
        self.residual_prior_encoder = ResidualPriorEncoder(config.num_features, hidden_size)

    def forward(
        self,
        base_prediction: torch.Tensor,
        inputs: Optional[torch.Tensor] = None,
        inputs_timestamps: Optional[torch.Tensor] = None,
    ) -> dict:
        """
        Shape:
            base_prediction: [B, O, F]
            inputs: [B, I, F] or None
            inputs_timestamps: [B, I, 2] or None
            return: dict[str, Tensor], each Tensor [B, O, F]
        """

        return {
            "horizon_condition": self.horizon_encoder(base_prediction, inputs_timestamps),
            "dynamic_condition": self.dynamic_encoder(inputs, base_prediction),
            "residual_prior": self.residual_prior_encoder(base_prediction, inputs),
        }
