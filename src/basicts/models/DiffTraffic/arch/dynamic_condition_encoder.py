from typing import Optional

import torch
from torch import nn


class DynamicConditionEncoder(nn.Module):
    """
    Encode short-term traffic dynamics that may explain structured residuals.
    """

    def __init__(self, num_features: int, hidden_size: int):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(num_features * 6, hidden_size),
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

        if inputs is None or inputs.size(1) < 2:
            return torch.zeros_like(base_prediction)

        deltas = inputs[:, 1:, :] - inputs[:, :-1, :]
        recent_window = min(deltas.size(1), 4)
        recent_deltas = deltas[:, -recent_window:, :]

        delta_mean = recent_deltas.mean(dim=1)
        delta_std = recent_deltas.std(dim=1, unbiased=False)
        last_delta = deltas[:, -1, :]
        slope = (inputs[:, -1, :] - inputs[:, 0, :]) / max(inputs.size(1) - 1, 1)
        centered_level = inputs[:, -1, :] - inputs.mean(dim=1)
        base_gap = base_prediction.mean(dim=1) - inputs[:, -1, :]

        summary = torch.cat(
            [delta_mean, delta_std, last_delta, slope, centered_level, base_gap],
            dim=-1,
        )
        encoded = self.proj(summary)
        return encoded.unsqueeze(1).expand(-1, base_prediction.size(1), -1).to(base_prediction.dtype)
