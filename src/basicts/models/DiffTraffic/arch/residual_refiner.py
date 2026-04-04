import torch
from torch import nn

from .residual_refinement_block import ResidualRefinementBlock


class ResidualRefiner(nn.Module):
    """
    Deterministic residual refiner for DiffTraffic-v1.
    """

    def __init__(self, config):
        super().__init__()
        self.num_features = config.num_features
        hidden_size = config.residual_hidden_size
        dropout = config.residual_dropout
        self.num_layers = max(int(getattr(config, "residual_refinement_layers", 2)), 1)

        self.condition_projection = nn.Sequential(
            nn.Linear(self.num_features * 4, hidden_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, self.num_features),
        )
        self.initializer = nn.Sequential(
            nn.Linear(self.num_features * 2, hidden_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, self.num_features),
        )
        self.refinement_blocks = nn.ModuleList(
            [ResidualRefinementBlock(self.num_features, hidden_size, dropout) for _ in range(self.num_layers)]
        )
        self.output_head = nn.Linear(self.num_features, self.num_features)

        nn.init.zeros_(self.initializer[-1].weight)
        nn.init.zeros_(self.initializer[-1].bias)
        nn.init.zeros_(self.output_head.weight)
        nn.init.zeros_(self.output_head.bias)

    def forward(
        self,
        base_prediction: torch.Tensor,
        condition: dict,
    ) -> dict:
        """
        Shape:
            base_prediction: [B, O, F]
            condition: dict with fused condition tensors, each [B, O, F] when present
            return:
                residual_prediction: [B, O, F]
        """

        temporal_condition = condition.get("temporal_condition", torch.zeros_like(base_prediction))
        dynamic_condition = condition.get("dynamic_condition", torch.zeros_like(base_prediction))
        graph_condition = condition.get("graph_condition", torch.zeros_like(base_prediction))
        residual_prior = condition.get("residual_prior", torch.zeros_like(base_prediction))

        condition_signal = self.condition_projection(
            torch.cat(
                [temporal_condition, dynamic_condition, graph_condition, residual_prior],
                dim=-1,
            )
        )
        residual_state = self.initializer(torch.cat([base_prediction, condition_signal], dim=-1))
        conditioned_context = condition_signal + residual_prior
        for block in self.refinement_blocks:
            residual_state = block(residual_state, conditioned_context)

        residual_prediction = self.output_head(residual_state)
        return {
            "residual_prediction": residual_prediction,
            "aux_info": {
                "refiner_layers": self.num_layers,
                "condition_keys": sorted(k for k, v in condition.items() if isinstance(v, torch.Tensor) and v is not None),
            },
        }
