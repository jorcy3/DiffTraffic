import torch

from .frequency_residual_encoder import FrequencyResidualEncoder


class FrequencyConditionEncoder(FrequencyResidualEncoder):
    """
    Frequency condition encoder alias with naming aligned to modular v1.5 design.
    """

    def forward(self, base_prediction: torch.Tensor) -> torch.Tensor:
        """
        Shape:
            base_prediction: [B, O, F]
            return: [B, O, F]
        """

        return super().forward(base_prediction)

