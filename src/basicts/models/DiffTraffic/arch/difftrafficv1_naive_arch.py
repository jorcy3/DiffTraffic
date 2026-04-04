from typing import Optional

import torch
from torch import nn

from ..config.difftrafficv1_config import DiffTrafficV1Config
from .naive_residual_head import NaiveResidualHead
from .staeformer_adapter import STAEformerAdapter


class DiffTrafficV1NaiveForForecasting(nn.Module):
    """
    Naive residual ablation built on top of the same STAEformer backbone.
    """

    def __init__(self, config: DiffTrafficV1Config):
        super().__init__()
        self.config = config
        self.adapter = STAEformerAdapter(config)
        self.naive_head = NaiveResidualHead(config)

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
                prediction: [B, O, F]
                base_prediction: [B, O, F]
                residual_prediction: [B, O, F]
        """

        base_outputs = self.adapter(
            inputs=inputs,
            inputs_timestamps=inputs_timestamps,
            step=step,
            epoch=epoch,
            train=train,
        )
        residual_outputs = self.naive_head(
            base_prediction=base_outputs["base_prediction"],
            inputs=inputs,
        )
        prediction = base_outputs["base_prediction"] + residual_outputs["residual_prediction"]
        return {
            "prediction": prediction,
            "base_prediction": base_outputs["base_prediction"],
            "residual_prediction": residual_outputs["residual_prediction"],
            "aux_info": {
                "adapter": base_outputs.get("aux_info", {}),
                "head": residual_outputs.get("aux_info", {}),
                "loss_weights": {
                    "pred": float(getattr(self.config, "loss_weight_pred", 1.0)),
                    "residual": float(getattr(self.config, "loss_weight_residual", 0.5)),
                    "diff": 0.0,
                    "freq": 0.0,
                },
            },
        }
