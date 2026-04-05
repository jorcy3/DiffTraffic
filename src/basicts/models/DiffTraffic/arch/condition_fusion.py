from typing import Any, Optional

import torch
from torch import nn


class ConditionFusion(nn.Module):
    """
    Organize and align encoded conditions for the residual decoder.

    Expected shapes:
        base_prediction: [batch_size, output_len, num_features]
        backbone_state: optional decoder-side state, implementation-defined
    """

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.use_backbone_state = bool(getattr(config, "use_backbone_state", False))

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
        horizon_condition = encoded_conditions.get("horizon_condition")
        temporal_condition = encoded_conditions.get("temporal_condition", torch.zeros_like(base_prediction))
        dynamic_condition = encoded_conditions.get("dynamic_condition", torch.zeros_like(base_prediction))
        graph_condition = encoded_conditions.get("graph_condition", torch.zeros_like(base_prediction))
        residual_prior = encoded_conditions.get("residual_prior")
        if residual_prior is not None:
            residual_prior = residual_prior.to(base_prediction.dtype)

        if horizon_condition is None:
            horizon_condition = temporal_condition
        else:
            horizon_condition = horizon_condition.to(base_prediction.dtype)

        condition_keys = ["horizon_condition", "dynamic_condition"]
        if graph_condition is not None and torch.count_nonzero(graph_condition).item() > 0:
            condition_keys.append("graph_condition")
        if residual_prior is not None:
            condition_keys.append("residual_prior")

        return {
            "base_prediction": base_prediction,
            "inputs": inputs,
            "backbone_state": fused_backbone_state,
            "inputs_timestamps": inputs_timestamps,
            "horizon_condition": horizon_condition,
            "temporal_condition": temporal_condition.to(base_prediction.dtype),
            "dynamic_condition": dynamic_condition.to(base_prediction.dtype),
            "graph_condition": graph_condition.to(base_prediction.dtype),
            "residual_prior": residual_prior,
            "condition_mode": "+".join(condition_keys),
            "condition_keys": condition_keys,
        }

