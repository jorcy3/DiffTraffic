import torch
from torch import nn


class FrequencyResidualEncoder(nn.Module):
    """
    Lightweight frequency-domain condition encoder.
    """

    def __init__(self, num_features: int, hidden_size: int):
        super().__init__()
        self.freq_mlp = nn.Sequential(
            nn.Linear(num_features, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, num_features),
        )

    def forward(self, residual_proxy: torch.Tensor) -> torch.Tensor:
        """
        Shape:
            residual_proxy: [B, O, F]
            return: [B, O, F]
        """

        freq = torch.fft.rfft(residual_proxy, dim=1)
        freq_amp = torch.abs(freq).mean(dim=1, keepdim=True)
        freq_amp = freq_amp.expand(-1, residual_proxy.size(1), -1)
        return self.freq_mlp(freq_amp.to(residual_proxy.dtype))

