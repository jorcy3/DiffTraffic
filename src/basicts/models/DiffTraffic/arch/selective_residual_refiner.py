import torch
from torch import nn

from .residual_refinement_block import ResidualRefinementBlock


class SelectiveResidualRefiner(nn.Module):
    """
    Residual refiner with a selective correction gate.
    """

    def __init__(self, config):
        super().__init__()
        self.num_features = config.num_features
        self.output_len = config.output_len
        self.refiner_mode = getattr(config, "residual_refiner_mode", "selective")
        hidden_size = config.residual_hidden_size
        gate_hidden_size = getattr(config, "gate_hidden_size", None) or hidden_size
        dropout = config.residual_dropout
        self.num_layers = max(int(getattr(config, "residual_refinement_layers", 2)), 1)

        self.condition_projection = nn.Sequential(
            nn.Linear(self.num_features * 3, hidden_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, self.num_features),
        )
        self.horizon_scale = nn.Parameter(torch.linspace(0.75, 1.25, steps=self.output_len).view(1, self.output_len, 1))
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
        self.gate_net = nn.Sequential(
            nn.Linear(self.num_features * 5, gate_hidden_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(gate_hidden_size, self.num_features),
        )
        gate_bias_init = float(getattr(config, "gate_bias_init", -2.0))

        nn.init.zeros_(self.initializer[-1].weight)
        nn.init.zeros_(self.initializer[-1].bias)
        nn.init.zeros_(self.output_head.weight)
        nn.init.zeros_(self.output_head.bias)
        nn.init.zeros_(self.gate_net[-1].weight)
        nn.init.constant_(self.gate_net[-1].bias, gate_bias_init)

    def forward(
        self,
        base_prediction: torch.Tensor,
        condition: dict,
    ) -> dict:
        """
        Shape:
            base_prediction: [B, O, F]
            condition: dict with fused condition tensors, each [B, O, F]
            return:
                residual_prediction: [B, O, F]
                residual_gate: [B, O, F]
        """

        horizon_condition = condition.get("horizon_condition", torch.zeros_like(base_prediction))
        dynamic_condition = condition.get("dynamic_condition", torch.zeros_like(base_prediction))
        residual_prior = condition.get("residual_prior", torch.zeros_like(base_prediction))

        condition_signal = self.condition_projection(
            torch.cat([horizon_condition, dynamic_condition, residual_prior], dim=-1)
        )
        conditioned_context = (condition_signal + residual_prior) * self.horizon_scale.to(base_prediction.dtype)

        if self.refiner_mode == "gate_only":
            residual_prediction = residual_prior
        else:
            residual_state = self.initializer(torch.cat([base_prediction, conditioned_context], dim=-1))
            for block in self.refinement_blocks:
                residual_state = block(residual_state, conditioned_context)
            residual_prediction = self.output_head(residual_state) * self.horizon_scale.to(base_prediction.dtype)

        gate_features = torch.cat(
            [
                base_prediction,
                horizon_condition,
                dynamic_condition,
                residual_prior,
                torch.abs(residual_prediction),
            ],
            dim=-1,
        )
        residual_gate = torch.sigmoid(self.gate_net(gate_features))
        effective_residual = residual_gate * residual_prediction
        return {
            "residual_prediction": residual_prediction,
            "residual_gate": residual_gate,
            "effective_residual_prediction": effective_residual,
            "aux_info": {
                "refiner_mode": self.refiner_mode,
                "refiner_layers": self.num_layers,
                "gate_mean": float(residual_gate.mean().item()),
                "condition_keys": sorted(k for k, v in condition.items() if isinstance(v, torch.Tensor) and v is not None),
            },
        }
