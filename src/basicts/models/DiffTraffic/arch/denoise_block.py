import torch
from torch import nn


class DenoiseBlock(nn.Module):
    """
    Small denoise block for residual diffusion decoding.
    """

    def __init__(self, num_features: int, hidden_size: int, dropout: float):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(num_features * 2 + 1, hidden_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, num_features),
        )

    def forward(
        self,
        noisy_residual: torch.Tensor,
        condition_signal: torch.Tensor,
        timestep_embedding: torch.Tensor,
    ) -> torch.Tensor:
        """
        Shape:
            noisy_residual: [B, O, F]
            condition_signal: [B, O, F]
            timestep_embedding: [B, O, 1]
            return: [B, O, F]
        """

        fused = torch.cat([noisy_residual, condition_signal, timestep_embedding], dim=-1)
        return self.net(fused)

