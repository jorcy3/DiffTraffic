from typing import Optional

import torch
from torch import nn


class TemporalConditionEncoder(nn.Module):
    """
    Encode future temporal context for each forecasting horizon.
    """

    def __init__(
        self,
        num_features: int,
        hidden_size: int,
        output_len: int,
        steps_per_day: int = 288,
        num_day_in_week: int = 7,
    ):
        super().__init__()
        self.output_len = output_len
        self.steps_per_day = steps_per_day
        self.num_day_in_week = num_day_in_week
        self.tod_embedding = nn.Embedding(steps_per_day, hidden_size)
        self.dow_embedding = nn.Embedding(num_day_in_week, hidden_size)
        self.horizon_embedding = nn.Embedding(output_len, hidden_size)
        self.proj = nn.Sequential(
            nn.Linear(hidden_size * 3, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, num_features),
        )

    def _recover_time_indices(self, inputs_timestamps: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        tod_index = torch.floor(inputs_timestamps[..., 0] * self.steps_per_day).long()
        tod_index = tod_index.clamp(min=0, max=self.steps_per_day - 1)

        if inputs_timestamps.size(-1) < 2:
            dow_index = torch.zeros_like(tod_index)
        else:
            dow_index = torch.floor(inputs_timestamps[..., 1] * self.num_day_in_week + 1e-6).long()
            dow_index = dow_index.clamp(min=0, max=self.num_day_in_week - 1)
        return tod_index, dow_index

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

        batch_size, output_len, _ = base_prediction.shape
        device = base_prediction.device

        future_offsets = torch.arange(1, output_len + 1, device=device).unsqueeze(0)
        if inputs_timestamps is None:
            last_tod = torch.zeros(batch_size, 1, dtype=torch.long, device=device)
            last_dow = torch.zeros(batch_size, 1, dtype=torch.long, device=device)
        else:
            tod_index, dow_index = self._recover_time_indices(inputs_timestamps)
            last_tod = tod_index[:, -1:].to(device)
            last_dow = dow_index[:, -1:].to(device)

        future_base = last_tod + future_offsets
        future_tod = future_base % self.steps_per_day
        future_dow = (last_dow + future_base // self.steps_per_day) % self.num_day_in_week
        horizon_index = torch.arange(output_len, device=device).unsqueeze(0).expand(batch_size, -1)

        temporal_feature = torch.cat(
            [
                self.tod_embedding(future_tod),
                self.dow_embedding(future_dow),
                self.horizon_embedding(horizon_index),
            ],
            dim=-1,
        )
        return self.proj(temporal_feature).to(base_prediction.dtype)
