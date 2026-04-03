import torch
from torch import nn


class ResidualRefinementBlock(nn.Module):
    """
    Conditioned residual refinement block with gated update.
    """

    def __init__(self, num_features: int, hidden_size: int, dropout: float):
        super().__init__()
        self.update_net = nn.Sequential(
            nn.Linear(num_features * 2, hidden_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, num_features),
        )
        self.gate_net = nn.Sequential(
            nn.Linear(num_features * 2, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, num_features),
            nn.Sigmoid(),
        )

    def forward(
        self,
        residual_state: torch.Tensor,
        conditioned_context: torch.Tensor,
    ) -> torch.Tensor:
        """
        Shape:
            residual_state: [B, O, F]
            conditioned_context: [B, O, F]
            return: [B, O, F]
        """

        fused = torch.cat([residual_state, conditioned_context], dim=-1)
        update = self.update_net(fused)
        gate = self.gate_net(fused)
        return residual_state + gate * update

