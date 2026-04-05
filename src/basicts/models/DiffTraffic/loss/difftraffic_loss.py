from typing import Optional

import torch


def _masked_l1(
    prediction: torch.Tensor,
    targets: torch.Tensor,
    targets_mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    if targets_mask is not None:
        diff = torch.abs(prediction - targets)
        diff = diff * targets_mask.to(diff.dtype)
        return diff.sum() / targets_mask.sum().clamp_min(1).to(diff.dtype)
    return torch.mean(torch.abs(prediction - targets))


def difftraffic_loss(
    prediction: torch.Tensor,
    targets: torch.Tensor,
    base_prediction: Optional[torch.Tensor] = None,
    residual_prediction: Optional[torch.Tensor] = None,
    residual_gate: Optional[torch.Tensor] = None,
    targets_mask: Optional[torch.Tensor] = None,
    aux_info: Optional[dict] = None,
    scaler_mean: Optional[torch.Tensor] = None,
    scaler_std: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Multi-term DiffTraffic loss with backward-compatible defaults.

    Terms:
        L_pred: supervised prediction loss
        L_residual: residual-consistency loss (optional)
        L_diff: diffusion regularization term (optional)
        L_freq: frequency-domain consistency (optional)
    """

    aux_info = aux_info or {}
    loss_weights = aux_info.get("loss_weights", {})
    w_pred = float(loss_weights.get("pred", 1.0))
    w_residual = float(loss_weights.get("residual", 0.0))
    w_gate = float(loss_weights.get("gate", 0.0))
    w_diff = float(loss_weights.get("diff", 0.0))
    w_freq = float(loss_weights.get("freq", 0.0))

    if scaler_mean is not None and scaler_std is not None:
        scaler_mean = scaler_mean.to(prediction.device)
        scaler_std = scaler_std.to(prediction.device)
        prediction = prediction * scaler_std + scaler_mean
        if base_prediction is not None:
            base_prediction = base_prediction * scaler_std + scaler_mean
        if residual_prediction is not None:
            residual_prediction = residual_prediction * scaler_std

    effective_residual = residual_prediction
    if effective_residual is not None and residual_gate is not None:
        effective_residual = residual_gate * effective_residual

    loss_pred = _masked_l1(prediction, targets, targets_mask)

    loss_total = w_pred * loss_pred

    if effective_residual is not None and base_prediction is not None and w_residual > 0.0:
        target_residual = targets - base_prediction.detach()
        loss_residual = _masked_l1(effective_residual, target_residual, targets_mask)
        loss_total = loss_total + w_residual * loss_residual

    if residual_gate is not None and w_gate > 0.0:
        loss_gate = residual_gate.mean()
        loss_total = loss_total + w_gate * loss_gate

    if effective_residual is not None and w_diff > 0.0:
        # Lightweight diffusion regularizer for residual smoothness.
        loss_diff = torch.mean(torch.abs(effective_residual[:, 1:, :] - effective_residual[:, :-1, :]))
        loss_total = loss_total + w_diff * loss_diff

    if effective_residual is not None and w_freq > 0.0:
        pred_freq = torch.abs(torch.fft.rfft(effective_residual, dim=1))
        target_proxy = targets - (base_prediction if base_prediction is not None else prediction - effective_residual)
        target_freq = torch.abs(torch.fft.rfft(target_proxy, dim=1))
        loss_freq = torch.mean(torch.abs(pred_freq - target_freq))
        loss_total = loss_total + w_freq * loss_freq

    return loss_total
