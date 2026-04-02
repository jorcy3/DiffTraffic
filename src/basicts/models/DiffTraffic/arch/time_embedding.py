from typing import Optional

import torch
from torch import nn


class SinusoidalTimeEmbedding(nn.Module):
    """
    Lightweight time or diffusion-step embedding.

    This is intentionally minimal and can be expanded later for adaptive noise or
    guidance conditioning.
    """

    def __init__(self, embedding_dim: int):
        super().__init__()
        self.embedding_dim = embedding_dim

    def forward(self, step: Optional[torch.Tensor] = None) -> Optional[torch.Tensor]:
        """
        Args:
            step: optional scalar or [batch_size] timestep tensor.

        Returns:
            Optional tensor with shape [batch_size, 1].
        """

        if step is None:
            return None
        return step.float().unsqueeze(-1)

