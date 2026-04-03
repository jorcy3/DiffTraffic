import torch

from basicts.metrics import masked_mae


def staeformer_official_loss(
    prediction: torch.Tensor,
    targets: torch.Tensor,
    targets_mask: torch.Tensor | None = None,
    scaler_mean: torch.Tensor | None = None,
    scaler_std: torch.Tensor | None = None,
) -> torch.Tensor:
    """
    Official-style STAEformer loss.

    Shape:
        prediction: [B, O, N]
        targets: [B, O, N]
        targets_mask: [B, O, N] or None
        scaler_mean: broadcastable to prediction or None
        scaler_std: broadcastable to prediction or None
        return: scalar tensor
    """

    if scaler_mean is not None and scaler_std is not None:
        scaler_mean = scaler_mean.to(prediction.device)
        scaler_std = scaler_std.to(prediction.device)
        prediction = prediction * scaler_std + scaler_mean

    return masked_mae(prediction, targets, targets_mask)
