from typing import Optional

import math
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
            Optional tensor with shape [batch_size, embedding_dim].
        """

        if step is None:
            return None
        if step.ndim == 0:
            step = step.unsqueeze(0)
        step = step.float()

        half_dim = self.embedding_dim // 2
        if half_dim == 0:
            return step.unsqueeze(-1)

        device = step.device
        exponents = torch.arange(half_dim, device=device, dtype=step.dtype)
        exponents = -math.log(10000.0) * exponents / max(half_dim - 1, 1)
        freqs = torch.exp(exponents)
        args = step.unsqueeze(-1) * freqs.unsqueeze(0)
        emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)

        if emb.size(-1) < self.embedding_dim:
            emb = torch.cat([emb, torch.zeros_like(emb[:, :1])], dim=-1)
        return emb

