from typing import Any, Optional

import torch
from torch import nn

from ..config import DiffTrafficV0Config


class ConditionFusion(nn.Module):
    """
    Prepare conditioning inputs for the residual decoder.

    Expected shapes:
        base_prediction: [batch_size, output_len, num_features]
        backbone_state: optional decoder-side state, implementation-defined
    """

    def __init__(self, config: DiffTrafficV0Config):
        super().__init__()
        self.config = config
        self.use_backbone_state = config.use_backbone_state or config.condition_mode == "base_and_state"

    def forward(
        self,
        base_prediction: torch.Tensor,
        backbone_state: Optional[Any] = None,
        inputs_timestamps: Optional[torch.Tensor] = None,
    ) -> dict:
        """
        Args:
            base_prediction: [batch_size, output_len, num_features]
            backbone_state: optional backbone context
            inputs_timestamps: [batch_size, input_len, num_timestamps]

        Returns:
            dict with:
                base_prediction: [batch_size, output_len, num_features]
                backbone_state: optional pass-through state.
                inputs_timestamps: [batch_size, input_len, num_timestamps]
                condition_mode: current conditioning mode.
        """

        fused_backbone_state = backbone_state if self.use_backbone_state else None
        return {
            "base_prediction": base_prediction,
            "backbone_state": fused_backbone_state,
            "inputs_timestamps": inputs_timestamps,
            "condition_mode": self.config.condition_mode,
        }

