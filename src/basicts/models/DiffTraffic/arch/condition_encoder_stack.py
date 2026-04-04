from typing import Optional

import torch
from torch import nn

from .dynamic_condition_encoder import DynamicConditionEncoder
from .graph_condition_encoder import GraphConditionEncoder
from .residual_prior_encoder import ResidualPriorEncoder
from .temporal_condition_encoder import TemporalConditionEncoder


class ConditionEncoderStack(nn.Module):
    """
    Independent condition encoder stack.
    """

    def __init__(self, config):
        super().__init__()
        self.config = config
        hidden_size = getattr(config, "condition_hidden_size", None) or config.residual_hidden_size
        graph_hidden_size = getattr(config, "graph_encoder_hidden_size", None) or hidden_size

        self.enable_graph_condition = bool(getattr(config, "enable_graph_condition", False))
        self.enable_residual_prior = bool(getattr(config, "enable_residual_prior", True))

        self.temporal_encoder = TemporalConditionEncoder(
            num_features=config.num_features,
            hidden_size=hidden_size,
            output_len=config.output_len,
            steps_per_day=getattr(config, "steps_per_day", 288),
            num_day_in_week=getattr(config, "num_day_in_week", 7),
        )
        self.dynamic_encoder = DynamicConditionEncoder(config.num_features, hidden_size)
        self.graph_encoder = GraphConditionEncoder(config.num_features, graph_hidden_size)
        self.residual_prior_encoder = ResidualPriorEncoder(config.num_features, hidden_size)

    def forward(
        self,
        base_prediction: torch.Tensor,
        inputs: Optional[torch.Tensor] = None,
        inputs_timestamps: Optional[torch.Tensor] = None,
        graph_prior: Optional[torch.Tensor] = None,
    ) -> dict:
        """
        Shape:
            base_prediction: [B, O, F]
            inputs: [B, I, F] or None
            inputs_timestamps: [B, I, T] or None
            graph_prior: [F, F] or [B, F, F] or None
            return: dict[str, Tensor|None], each Tensor shape [B, O, F]
        """

        temporal_condition = self.temporal_encoder(base_prediction, inputs_timestamps)
        dynamic_condition = self.dynamic_encoder(inputs, base_prediction)

        if self.enable_graph_condition:
            graph_condition = self.graph_encoder(base_prediction, graph_prior)
        else:
            graph_condition = torch.zeros_like(base_prediction)

        if self.enable_residual_prior:
            residual_prior = self.residual_prior_encoder(base_prediction, inputs)
        else:
            residual_prior = None

        return {
            "temporal_condition": temporal_condition,
            "dynamic_condition": dynamic_condition,
            "graph_condition": graph_condition,
            "residual_prior": residual_prior,
        }
