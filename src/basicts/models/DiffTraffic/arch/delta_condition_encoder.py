from typing import Optional

import torch
from torch import nn


class DeltaConditionEncoder(nn.Module):
    """
    Encode short-term dynamics from recent differenced inputs.
    """

    def __init__(self, num_features: int, hidden_size: int):
        super().__init__()
        self.num_features = num_features
        self.proj = nn.Sequential(
            nn.Linear(num_features * 3, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, num_features),
        )

    def forward(
        self,
        inputs: Optional[torch.Tensor],
        base_prediction: torch.Tensor,
    ) -> torch.Tensor:
        """
        Shape:
            inputs: [B, I, F] or None
            base_prediction: [B, O, F]
            return: [B, O, F]
        """

        if inputs is None:
            return torch.zeros_like(base_prediction)

        deltas = inputs[:, 1:, :] - inputs[:, :-1, :]
        delta_mean = deltas.mean(dim=1)
        delta_std = deltas.std(dim=1, unbiased=False)
        slope = (inputs[:, -1, :] - inputs[:, 0, :]) / max(inputs.size(1) - 1, 1)
        features = torch.cat([delta_mean, delta_std, slope], dim=-1)
        encoded = self.proj(features).unsqueeze(1).expand(-1, base_prediction.size(1), -1)
        return encoded.to(base_prediction.dtype)

