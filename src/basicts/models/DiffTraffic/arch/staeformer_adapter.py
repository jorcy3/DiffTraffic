from typing import Optional

import torch
from torch import nn

from basicts.models.STAEformer import STAEformer


class STAEformerAdapter(nn.Module):
    """
    Thin adapter that exposes STAEformer as the deterministic DiffTraffic base.
    """

    def __init__(self, config):
        super().__init__()
        self.backbone = STAEformer(config)

    def forward(
        self,
        inputs: torch.Tensor,
        inputs_timestamps: Optional[torch.Tensor] = None,
        step: Optional[int] = None,
        epoch: Optional[int] = None,
        train: Optional[bool] = None,
    ) -> dict:
        """
        Shape:
            inputs: [B, I, F]
            inputs_timestamps: [B, I, 2] or None
            return:
                base_prediction: [B, O, F]
        """

        base_prediction = self.backbone(inputs=inputs, inputs_timestamps=inputs_timestamps)
        return {
            "base_prediction": base_prediction,
            "backbone_state": None,
            "aux_info": {
                "backbone": self.backbone.__class__.__name__,
                "step": step,
                "epoch": epoch,
                "train": train,
            },
        }
