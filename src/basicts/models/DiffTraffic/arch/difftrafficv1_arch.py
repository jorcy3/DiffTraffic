from typing import Optional

import torch
from torch import nn

from ..config.difftrafficv1_config import DiffTrafficV1Config
from .condition_encoder_stack import ConditionEncoderStack
from .condition_fusion import ConditionFusion
from .residual_refiner import ResidualRefiner
from .staeformer_adapter import STAEformerAdapter


class DiffTrafficV1ForForecasting(nn.Module):
    """
    DiffTraffic-v1 forecasting model.

    Flow:
        inputs -> STAEformerAdapter -> base_prediction
        inputs/base_prediction -> ConditionEncoderStack -> encoded conditions
        encoded conditions -> ConditionFusion -> fused condition
        base_prediction/fused condition -> ResidualRefiner -> residual_prediction
        prediction = base_prediction + residual_prediction
    """

    def __init__(self, config: DiffTrafficV1Config):
        super().__init__()
        self.config = config
        self.adapter = STAEformerAdapter(config)
        self.condition_encoders = ConditionEncoderStack(config)
        self.condition_fusion = ConditionFusion(config)
        self.residual_refiner = ResidualRefiner(config)

    def forward(
        self,
        inputs: torch.Tensor,
        inputs_timestamps: Optional[torch.Tensor] = None,
        graph_prior: Optional[torch.Tensor] = None,
        step: Optional[int] = None,
        epoch: Optional[int] = None,
        train: Optional[bool] = None,
    ) -> dict:
        """
        Shape:
            inputs: [B, I, F]
            inputs_timestamps: [B, I, 2] or None
            graph_prior: [F, F] or [B, F, F] or None
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
        encoded_conditions = self.condition_encoders(
            base_prediction=base_outputs["base_prediction"],
            inputs=inputs,
            inputs_timestamps=inputs_timestamps,
            graph_prior=graph_prior,
        )
        fused_condition = self.condition_fusion(
            base_prediction=base_outputs["base_prediction"],
            encoded_conditions=encoded_conditions,
            inputs=inputs,
            backbone_state=base_outputs.get("backbone_state"),
            inputs_timestamps=inputs_timestamps,
        )
        residual_outputs = self.residual_refiner(
            base_prediction=base_outputs["base_prediction"],
            condition=fused_condition,
        )
        prediction = base_outputs["base_prediction"] + residual_outputs["residual_prediction"]

        return {
            "prediction": prediction,
            "base_prediction": base_outputs["base_prediction"],
            "residual_prediction": residual_outputs["residual_prediction"],
            "aux_info": {
                "adapter": base_outputs.get("aux_info", {}),
                "condition": {
                    "condition_mode": fused_condition.get("condition_mode"),
                    "condition_keys": fused_condition.get("condition_keys", []),
                },
                "refiner": residual_outputs.get("aux_info", {}),
                "loss_weights": {
                    "pred": float(getattr(self.config, "loss_weight_pred", 1.0)),
                    "residual": float(getattr(self.config, "loss_weight_residual", 0.5)),
                    "diff": 0.0,
                    "freq": 0.0,
                },
            },
        }
