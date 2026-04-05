from typing import Optional

import torch
from torch import nn

from ..config.difftrafficv11_config import DiffTrafficV11Config
from .condition_fusion import ConditionFusion
from .selective_condition_encoder_stack import SelectiveConditionEncoderStack
from .selective_residual_refiner import SelectiveResidualRefiner
from .staeformer_adapter import STAEformerAdapter


class DiffTrafficV11ForForecasting(nn.Module):
    """
    DiffTraffic-v1.1 with selective residual refinement.
    """

    def __init__(self, config: DiffTrafficV11Config):
        super().__init__()
        self.config = config
        self.adapter = STAEformerAdapter(config)
        self.condition_encoders = SelectiveConditionEncoderStack(config)
        self.condition_fusion = ConditionFusion(config)
        self.residual_refiner = SelectiveResidualRefiner(config)

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
                residual_gate: [B, O, F]
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
        effective_residual = residual_outputs["effective_residual_prediction"]
        prediction = base_outputs["base_prediction"] + effective_residual
        return {
            "prediction": prediction,
            "base_prediction": base_outputs["base_prediction"],
            "residual_prediction": residual_outputs["residual_prediction"],
            "residual_gate": residual_outputs["residual_gate"],
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
                    "gate": float(getattr(self.config, "loss_weight_gate", 0.01)),
                    "diff": 0.0,
                    "freq": 0.0,
                },
            },
        }
