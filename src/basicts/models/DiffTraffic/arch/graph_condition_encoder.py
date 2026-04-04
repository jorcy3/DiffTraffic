from typing import Optional

import torch

from .graph_delay_encoder import GraphDelayEncoder


class GraphConditionEncoder(GraphDelayEncoder):
    """
    Graph condition encoder alias with naming aligned to modular v1.5 design.
    """

    def forward(
        self,
        base_prediction: torch.Tensor,
        graph_prior: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Shape:
            base_prediction: [B, O, F]
            graph_prior: [F, F] or [B, F, F] or None
            return: [B, O, F]
        """

        return super().forward(base_prediction, graph_prior)
