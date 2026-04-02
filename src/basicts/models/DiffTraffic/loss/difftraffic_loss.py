from typing import Optional

import torch


def difftraffic_loss(
    prediction: torch.Tensor,
    targets: torch.Tensor,
    base_prediction: Optional[torch.Tensor] = None,
    residual_prediction: Optional[torch.Tensor] = None,
    targets_mask: Optional[torch.Tensor] = None,
    aux_info: Optional[dict] = None,
) -> torch.Tensor:
    """
    Minimal loss shell for DiffTraffic-v0.

    v0 uses the main supervised signal only. Extra arguments are kept for future
    residual/noise/guidance terms.
    """

    if targets_mask is not None:
        diff = torch.abs(prediction - targets)
        diff = diff * targets_mask.to(diff.dtype)
        return diff.sum() / targets_mask.sum().clamp_min(1).to(diff.dtype)
    return torch.mean(torch.abs(prediction - targets))
