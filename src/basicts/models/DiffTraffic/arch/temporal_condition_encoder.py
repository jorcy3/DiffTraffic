from typing import Optional

import torch
from torch import nn


class TemporalConditionEncoder(nn.Module):
    """
    Encode temporal condition for each forecasting horizon.
    """

    def __init__(self, num_features: int, hidden_size: int):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(num_features, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, num_features),
        )

    def forward(
        self,
        base_prediction: torch.Tensor,
        inputs_timestamps: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Shape:
            base_prediction: [B, O, F]
            inputs_timestamps: [B, I, T] or None
            return: [B, O, F]
        """

        temporal = self.proj(base_prediction)
        if inputs_timestamps is None:
            return temporal
        ts_scale = inputs_timestamps.mean(dim=(1, 2), keepdim=True).to(base_prediction.dtype)
        return temporal + ts_scale

