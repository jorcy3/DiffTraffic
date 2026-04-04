import torch
from torch import nn


class NaiveResidualHead(nn.Module):
    """
    Simple residual head for ablation against the structured v1 refiner.
    """

    def __init__(self, config):
        super().__init__()
        hidden_size = config.residual_hidden_size
        dropout = config.residual_dropout
        self.net = nn.Sequential(
            nn.Linear(config.num_features * 2, hidden_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, config.num_features),
        )
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(
        self,
        base_prediction: torch.Tensor,
        inputs: torch.Tensor | None = None,
    ) -> dict:
        """
        Shape:
            base_prediction: [B, O, F]
            inputs: [B, I, F] or None
            return:
                residual_prediction: [B, O, F]
        """

        if inputs is None:
            last_input = torch.zeros_like(base_prediction)
        else:
            last_input = inputs[:, -1:, :].expand(-1, base_prediction.size(1), -1)
        residual_prediction = self.net(torch.cat([base_prediction, last_input], dim=-1))
        return {
            "residual_prediction": residual_prediction,
            "aux_info": {
                "head": self.__class__.__name__,
            },
        }
