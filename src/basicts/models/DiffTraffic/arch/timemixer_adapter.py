from typing import Optional

import torch
from torch import nn

from basicts.models.TimeMixer import TimeMixerForForecasting

from ..config import DiffTrafficV0Config


class TimeMixerAdapter(nn.Module):
    """
    Thin wrapper over the existing TimeMixer implementation.

    v0 responsibility:
    - reuse TimeMixer as the base forecaster
    - expose a stable output contract for DiffTraffic
    """

    def __init__(self, config: DiffTrafficV0Config):
        super().__init__()
        self.config = config
        self.base_model = TimeMixerForForecasting(config)
        self.use_backbone_state = config.use_backbone_state or config.condition_mode == "base_and_state"

    def forward(
        self,
        inputs: torch.Tensor,
        inputs_timestamps: Optional[torch.Tensor] = None,
        step: Optional[int] = None,
        epoch: Optional[int] = None,
        train: Optional[bool] = None,
    ) -> dict:
        """
        Args:
            inputs: [batch_size, input_len, num_features]
            inputs_timestamps: [batch_size, input_len, num_timestamps]

        Returns:
            dict with:
                base_prediction: [batch_size, output_len, num_features]
                backbone_state: optional pass-through state.
                aux_info: auxiliary metadata.
        """
        # TimeMixer's multi-scale path expects timestamp tensors to be indexable.
        # Keep v0 robust by providing a minimal placeholder when timestamps are absent.
        if inputs_timestamps is None:
            batch_size, input_len, _ = inputs.shape
            inputs_timestamps = torch.zeros(
                batch_size,
                input_len,
                1,
                dtype=inputs.dtype,
                device=inputs.device,
            )

        base_prediction = self.base_model(inputs, inputs_timestamps)
        backbone_state = None
        if self.use_backbone_state:
            backbone_state = {
                "step": step,
                "epoch": epoch,
                "train": train,
            }
        return {
            "base_prediction": base_prediction,
            "backbone_state": backbone_state,
            "aux_info": {
                "step": step,
                "epoch": epoch,
                "train": train,
            },
        }

