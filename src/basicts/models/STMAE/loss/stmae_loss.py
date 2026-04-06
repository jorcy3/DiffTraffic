from __future__ import annotations

import torch

from basicts.metrics import masked_mae


def stmae_style_loss(
    prediction: torch.Tensor,
    targets: torch.Tensor,
    targets_mask: torch.Tensor | None = None,
    scaler_mean: torch.Tensor | None = None,
    scaler_std: torch.Tensor | None = None,
    masked_reconstruction: torch.Tensor | None = None,
    masked_positions: torch.Tensor | None = None,
    enhancement_targets: torch.Tensor | None = None,
    enhancement_valid_mask: torch.Tensor | None = None,
    aux_info: dict | None = None,
) -> torch.Tensor:
    """
    Forecasting loss plus masked reconstruction loss for STMAE-style enhancement.

    Shape:
        prediction: [B, O, N]
        targets: [B, O, N]
        masked_reconstruction: [B, I, N] or None
        masked_positions: [B, I, N] or None
        enhancement_targets: [B, I, N] or None
        return: scalar
    """

    aux_info = aux_info or {}
    loss_weights = aux_info.get("loss_weights", {})
    w_prediction = float(loss_weights.get("prediction", 1.0))
    w_reconstruction = float(loss_weights.get("reconstruction", 0.0))

    if scaler_mean is not None and scaler_std is not None:
        scaler_mean = scaler_mean.to(prediction.device)
        scaler_std = scaler_std.to(prediction.device)
        prediction = prediction * scaler_std + scaler_mean

    loss = w_prediction * masked_mae(prediction, targets, targets_mask)

    if (
        masked_reconstruction is not None
        and masked_positions is not None
        and enhancement_targets is not None
        and w_reconstruction > 0.0
    ):
        recon_mask = masked_positions.to(torch.bool)
        if enhancement_valid_mask is not None:
            recon_mask = recon_mask & enhancement_valid_mask.to(torch.bool)
        if torch.any(recon_mask):
            recon_error = torch.abs(masked_reconstruction - enhancement_targets)
            recon_error = recon_error * recon_mask.to(recon_error.dtype)
            recon_loss = recon_error.sum() / recon_mask.sum().clamp_min(1).to(recon_error.dtype)
            loss = loss + w_reconstruction * recon_loss

    return loss
