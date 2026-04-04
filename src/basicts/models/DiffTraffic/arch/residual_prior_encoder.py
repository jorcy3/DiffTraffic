from typing import Optional

import torch
from torch import nn


class ResidualPriorEncoder(nn.Module):
    """
    Encode a residual prior that highlights likely correction regions.
    """

    def __init__(self, num_features: int, hidden_size: int):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(num_features * 4, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, num_features),
        )

    def forward(
        self,
        base_prediction: torch.Tensor,
        inputs: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Shape:
            base_prediction: [B, O, F]
            inputs: [B, I, F] or None
            return: [B, O, F]
        """

        base_level = base_prediction.mean(dim=1)
        base_spread = base_prediction.std(dim=1, unbiased=False)
        if inputs is None:
            volatility = torch.zeros_like(base_level)
            anchor_gap = torch.zeros_like(base_level)
        else:
            volatility = (inputs[:, 1:, :] - inputs[:, :-1, :]).abs().mean(dim=1)
            anchor_gap = base_prediction[:, 0, :] - inputs[:, -1, :]
        prior = self.proj(torch.cat([base_level, base_spread, volatility, anchor_gap], dim=-1))
        return prior.unsqueeze(1).expand(-1, base_prediction.size(1), -1).to(base_prediction.dtype)
