from __future__ import annotations

from typing import Dict

import torch
from torch.utils.data import DataLoader

from basicts.data import BasicTSForecastingDataset
from basicts.utils.constants import BasicTSMode

from .arch import DiffTrafficV0ForForecasting
from .config import DiffTrafficV0Config
from .loss import difftraffic_loss


def build_smoke_config() -> DiffTrafficV0Config:
    """
    Build a compact DiffTraffic-v0 config for import and shape checks.
    """

    return DiffTrafficV0Config(
        input_len=16,
        output_len=8,
        num_features=3,
        hidden_size=16,
        num_layers=1,
        down_sampling_layers=1,
        down_sampling_window=2,
        moving_avg=3,
        residual_hidden_size=8,
        residual_dropout=0.0,
        use_timestamps=False,
        timestamp_sizes=None,
    )


def check_config_instantiation() -> DiffTrafficV0Config:
    """
    Instantiate the minimal config and return it for downstream checks.
    """

    config = build_smoke_config()
    _ = config.input_len
    _ = config.output_len
    _ = config.num_features
    return config


def check_prediction_shapes(
    prediction: torch.Tensor,
    base_prediction: torch.Tensor,
    residual_prediction: torch.Tensor,
    targets: torch.Tensor,
) -> Dict[str, tuple[int, ...]]:
    """
    Validate that the model outputs align with the forecasting target shape.
    """

    expected_shape = tuple(targets.shape)
    shapes = {
        "prediction": tuple(prediction.shape),
        "base_prediction": tuple(base_prediction.shape),
        "residual_prediction": tuple(residual_prediction.shape),
        "targets": expected_shape,
    }
    if shapes["prediction"] != expected_shape:
        raise ValueError(f"prediction shape {shapes['prediction']} does not match targets shape {expected_shape}.")
    if shapes["base_prediction"] != expected_shape:
        raise ValueError(
            f"base_prediction shape {shapes['base_prediction']} does not match targets shape {expected_shape}."
        )
    if shapes["residual_prediction"] != expected_shape:
        raise ValueError(
            f"residual_prediction shape {shapes['residual_prediction']} does not match targets shape {expected_shape}."
        )
    return shapes


def check_loss_call(
    prediction: torch.Tensor,
    targets: torch.Tensor,
    base_prediction: torch.Tensor,
    residual_prediction: torch.Tensor,
) -> torch.Tensor:
    """
    Run the minimal supervised loss call used by the v0 shell.
    """

    loss = difftraffic_loss(
        prediction=prediction,
        targets=targets,
        base_prediction=base_prediction,
        residual_prediction=residual_prediction,
    )
    if loss.ndim != 0:
        raise ValueError(f"Expected scalar loss, got shape {tuple(loss.shape)}.")
    return loss


def run_minimal_checks() -> Dict[str, object]:
    """
    Run config, forward, shape, and loss smoke checks for DiffTraffic-v0.
    """

    config = check_config_instantiation()
    model = DiffTrafficV0ForForecasting(config)
    inputs = torch.randn(2, config.input_len, config.num_features)
    inputs_timestamps = torch.zeros(2, config.input_len, 1)
    targets = torch.randn(2, config.output_len, config.num_features)

    outputs = model(inputs=inputs, inputs_timestamps=inputs_timestamps)
    shapes = check_prediction_shapes(
        prediction=outputs["prediction"],
        base_prediction=outputs["base_prediction"],
        residual_prediction=outputs["residual_prediction"],
        targets=targets,
    )
    loss = check_loss_call(
        prediction=outputs["prediction"],
        targets=targets,
        base_prediction=outputs["base_prediction"],
        residual_prediction=outputs["residual_prediction"],
    )
    return {
        "config": config,
        "model": model.__class__.__name__,
        "shapes": shapes,
        "loss": float(loss.item()),
    }


def build_metrla_real_step_config() -> DiffTrafficV0Config:
    """
    Build a small DiffTraffic-v0 config that matches the real METR-LA dataset.
    """

    return DiffTrafficV0Config(
        input_len=12,
        output_len=12,
        num_features=207,
        hidden_size=32,
        num_layers=1,
        down_sampling_layers=1,
        down_sampling_window=2,
        moving_avg=3,
        use_timestamps=True,
        timestamp_sizes=[288, 7],
        residual_hidden_size=64,
        residual_dropout=0.0,
    )


def run_real_metrla_train_step(batch_size: int = 2) -> Dict[str, object]:
    """
    Run one real METR-LA training step with the existing DiffTraffic-v0 shell.
    """

    config = build_metrla_real_step_config()
    dataset = BasicTSForecastingDataset(
        dataset_name="METR-LA",
        input_len=config.input_len,
        output_len=config.output_len,
        mode=BasicTSMode.TRAIN,
        use_timestamps=True,
        local=True,
        data_file_path="datasets/METR-LA",
        memmap=False,
    )
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    batch = next(iter(loader))

    model = DiffTrafficV0ForForecasting(config)
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    optimizer.zero_grad()

    inputs = batch["inputs"].float()
    inputs_timestamps = batch["inputs_timestamps"].float()
    targets = batch["targets"].float()

    outputs = model(inputs=inputs, inputs_timestamps=inputs_timestamps)
    shapes = check_prediction_shapes(
        prediction=outputs["prediction"],
        base_prediction=outputs["base_prediction"],
        residual_prediction=outputs["residual_prediction"],
        targets=targets,
    )
    loss = check_loss_call(
        prediction=outputs["prediction"],
        targets=targets,
        base_prediction=outputs["base_prediction"],
        residual_prediction=outputs["residual_prediction"],
    )
    loss.backward()
    optimizer.step()

    return {
        "dataset_batch_shapes": {
            "inputs": tuple(batch["inputs"].shape),
            "inputs_timestamps": tuple(batch["inputs_timestamps"].shape),
            "targets": tuple(batch["targets"].shape),
            "targets_timestamps": tuple(batch["targets_timestamps"].shape),
        },
        "model_io_shapes": {
            "prediction": tuple(outputs["prediction"].shape),
            "base_prediction": tuple(outputs["base_prediction"].shape),
            "residual_prediction": tuple(outputs["residual_prediction"].shape),
            "targets": tuple(targets.shape),
        },
        "dtypes": {
            "inputs": str(inputs.dtype),
            "inputs_timestamps": str(inputs_timestamps.dtype),
            "targets": str(targets.dtype),
            "prediction": str(outputs["prediction"].dtype),
        },
        "devices": {
            "inputs": str(inputs.device),
            "inputs_timestamps": str(inputs_timestamps.device),
            "targets": str(targets.device),
            "prediction": str(outputs["prediction"].device),
        },
        "loss": float(loss.item()),
        "shapes": shapes,
    }


