from typing import Any, Optional

import torch
from torch import nn

from ..config import DiffTrafficV0Config


class ConditionFusion(nn.Module):
    """
    Organize and align encoded conditions for the residual decoder.

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
        encoded_conditions: Optional[dict] = None,
        inputs: Optional[torch.Tensor] = None,
        backbone_state: Optional[Any] = None,
        inputs_timestamps: Optional[torch.Tensor] = None,
    ) -> dict:
        """
        Args:
            base_prediction: [batch_size, output_len, num_features]
            encoded_conditions: optional dict from ConditionEncoderStack
            inputs: [batch_size, input_len, num_features]
            backbone_state: optional backbone context
            inputs_timestamps: [batch_size, input_len, num_timestamps]

        Returns:
            dict with:
                base_prediction: [batch_size, output_len, num_features]
                backbone_state: optional pass-through state.
                inputs: [batch_size, input_len, num_features]
                inputs_timestamps: [batch_size, input_len, num_timestamps]
                temporal_condition: [batch_size, output_len, num_features]
                delta_condition: [batch_size, output_len, num_features]
                graph_condition: [batch_size, output_len, num_features]
                frequency_condition: [batch_size, output_len, num_features] or None
                condition_mode: current conditioning mode.
        """

        fused_backbone_state = backbone_state if self.use_backbone_state else None
        encoded_conditions = encoded_conditions or {}
        temporal_condition = encoded_conditions.get("temporal_condition", torch.zeros_like(base_prediction))
        delta_condition = encoded_conditions.get("delta_condition", torch.zeros_like(base_prediction))
        graph_condition = encoded_conditions.get("graph_condition", torch.zeros_like(base_prediction))
        frequency_condition = encoded_conditions.get("frequency_condition")
        residual_prior = encoded_conditions.get("residual_prior")

        if frequency_condition is not None:
            frequency_condition = frequency_condition.to(base_prediction.dtype)
        if residual_prior is not None:
            residual_prior = residual_prior.to(base_prediction.dtype)

        return {
            "base_prediction": base_prediction,
            "inputs": inputs,
            "backbone_state": fused_backbone_state,
            "inputs_timestamps": inputs_timestamps,
            "temporal_condition": temporal_condition.to(base_prediction.dtype),
            "delta_condition": delta_condition.to(base_prediction.dtype),
            "graph_condition": graph_condition.to(base_prediction.dtype),
            "frequency_condition": frequency_condition,
            "residual_prior": residual_prior,
            "condition_mode": self.config.condition_mode,
        }

