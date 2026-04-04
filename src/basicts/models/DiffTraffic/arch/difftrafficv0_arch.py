from typing import Optional

import torch
from torch import nn

from ..config.difftrafficv0_config import DiffTrafficV0Config
from .condition_encoder_stack import ConditionEncoderStack
from .condition_fusion import ConditionFusion
from .residual_diffusion_decoder import ResidualDiffusionDecoder
from .timemixer_adapter import TimeMixerAdapter


class DiffTrafficV0ForForecasting(nn.Module):
    """
    DiffTraffic-v0 forecasting model.

    Flow:
        inputs -> TimeMixerAdapter -> Y_base
        Y_base -> ConditionFusion -> decoder condition
        condition -> ResidualDiffusionDecoder -> R_hat
        Y_hat = Y_base + R_hat
    """

    def __init__(self, config: DiffTrafficV0Config):
        super().__init__()
        self.config = config
        self.adapter = TimeMixerAdapter(config)
        self.condition_encoders = ConditionEncoderStack(config)
        self.condition_fusion = ConditionFusion(config)
        self.residual_decoder = ResidualDiffusionDecoder(config)

    def forward(
        self,
        inputs: torch.Tensor,
        inputs_timestamps: Optional[torch.Tensor] = None,
        step: Optional[int] = None,
        epoch: Optional[int] = None,
        train: Optional[bool] = None,
    ) -> dict:
        """
        Args:
            inputs: [batch_size, input_len, num_features]
            inputs_timestamps: [batch_size, input_len, num_timestamps]

        Returns:
            dict with:
                prediction: [batch_size, output_len, num_features]
                base_prediction: [batch_size, output_len, num_features]
                residual_prediction: [batch_size, output_len, num_features]
                aux_info: auxiliary metadata.
        """

        base_outputs = self.adapter(
            inputs=inputs,
            inputs_timestamps=inputs_timestamps,
            step=step,
            epoch=epoch,
            train=train,
        )
        condition = self.condition_fusion(
            base_prediction=base_outputs["base_prediction"],
            encoded_conditions=self.condition_encoders(
                base_prediction=base_outputs["base_prediction"],
                inputs=inputs,
                inputs_timestamps=inputs_timestamps,
            ),
            inputs=inputs,
            backbone_state=base_outputs.get("backbone_state"),
            inputs_timestamps=inputs_timestamps,
        )
        residual_outputs = self.residual_decoder(
            condition=condition,
            step=step,
            epoch=epoch,
            train=train,
        )
        prediction = base_outputs["base_prediction"] + residual_outputs["residual_prediction"]
        return {
            "prediction": prediction,
            "base_prediction": base_outputs["base_prediction"],
            "residual_prediction": residual_outputs["residual_prediction"],
            "aux_info": {
                "adapter": base_outputs.get("aux_info", {}),
                "decoder": residual_outputs.get("aux_info", {}),
                "condition_mode": condition.get("condition_mode"),
                "has_backbone_state": condition.get("backbone_state") is not None,
                "condition_keys": list(condition.keys()),
                "loss_weights": {
                    "pred": float(getattr(self.config, "loss_weight_pred", 1.0)),
                    "residual": float(getattr(self.config, "loss_weight_residual", 0.0)),
                    "diff": float(getattr(self.config, "loss_weight_diff", 0.0)),
                    "freq": float(getattr(self.config, "loss_weight_freq", 0.0)),
                },
            },
        }

