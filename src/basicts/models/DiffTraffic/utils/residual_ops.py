import torch


def build_residual_target(targets: torch.Tensor, base_prediction: torch.Tensor) -> torch.Tensor:
    """
    Compute the residual target used by the decoder.

    Shapes:
        targets: [batch_size, output_len, num_features]
        base_prediction: [batch_size, output_len, num_features]
    """

    return targets - base_prediction


def merge_prediction(base_prediction: torch.Tensor, residual_prediction: torch.Tensor) -> torch.Tensor:
    """
    Merge the base prediction with the residual prediction.
    """

    return base_prediction + residual_prediction


def zero_residual_like(base_prediction: torch.Tensor) -> torch.Tensor:
    """
    Create a zero residual tensor matching the base prediction shape.
    """

    return torch.zeros_like(base_prediction)
