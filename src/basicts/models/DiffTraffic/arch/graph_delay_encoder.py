from typing import Optional

import torch
from torch import nn


class GraphDelayEncoder(nn.Module):
    """
    Optional graph-prior encoder with safe fallback.
    """

    def __init__(self, num_features: int, hidden_size: int):
        super().__init__()
        self.num_features = num_features
        self.graph_mlp = nn.Sequential(
            nn.Linear(num_features, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, num_features),
        )

    def forward(
        self,
        base_prediction: torch.Tensor,
        graph_prior: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Shape:
            base_prediction: [B, O, F]
            graph_prior: [F, F] or [B, F, F] or None
            return: [B, O, F]
        """

        if graph_prior is None:
            return torch.zeros_like(base_prediction)

        if graph_prior.ndim == 2:
            graph_prior = graph_prior.unsqueeze(0)
        graph_prior = graph_prior.to(base_prediction.device, dtype=base_prediction.dtype)

        signal = base_prediction.mean(dim=1)  # [B, F]
        if graph_prior.size(0) == 1:
            mixed = torch.matmul(signal, graph_prior[0])
        else:
            mixed = torch.bmm(signal.unsqueeze(1), graph_prior).squeeze(1)
        mixed = mixed.unsqueeze(1).expand(-1, base_prediction.size(1), -1)
        return self.graph_mlp(mixed)

