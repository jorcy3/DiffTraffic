from __future__ import annotations

from types import SimpleNamespace
from typing import Dict

import torch
from torch.utils.data import DataLoader

from basicts.data import BasicTSForecastingDataset
from basicts.runners.builder import Builder
from basicts.utils.constants import BasicTSMode

from .arch import STAEformerGraph
from .config import STAEformerGraphConfig, build_official_staeformer_graph_forecasting_config


def build_smoke_config() -> STAEformerGraphConfig:
    """
    Build a compact config for import and shape checks.
    """

    return STAEformerGraphConfig(
        input_len=8,
        output_len=4,
        num_features=3,
        input_embedding_dim=4,
        tod_embedding_dim=4,
        dow_embedding_dim=4,
        adaptive_embedding_dim=4,
        feed_forward_dim=16,
        num_heads=1,
        num_layers=1,
        dropout=0.0,
        graph_bias_enabled=False,
    )


def run_minimal_checks() -> Dict[str, object]:
    """
    Run config, forward, and shape smoke checks for STAEformerGraph.
    """

    config = build_smoke_config()
    model = STAEformerGraph(config)
    inputs = torch.randn(2, config.input_len, config.num_features)
    inputs_timestamps = torch.zeros(2, config.input_len, 2)

    prediction = model(inputs=inputs, inputs_timestamps=inputs_timestamps)
    if tuple(prediction.shape) != (2, config.output_len, config.num_features):
        raise ValueError(f"Unexpected prediction shape: {tuple(prediction.shape)}")

    return {
        "model": model.__class__.__name__,
        "prediction_shape": tuple(prediction.shape),
        "graph_bias_enabled": config.graph_bias_enabled,
    }


def run_real_metrla_train_step(batch_size: int = 2) -> Dict[str, object]:
    """
    Run one real METR-LA training step with STAEformerGraph.
    """

    cfg = build_official_staeformer_graph_forecasting_config(
        dataset_name="METR-LA",
        data_file_path="datasets/METR-LA",
        graph_hop_radius=2,
        graph_bias_init=0.20,
        graph_far_bias=0.0,
    )
    dataset = BasicTSForecastingDataset(
        dataset_name="METR-LA",
        input_len=cfg.input_len,
        output_len=cfg.output_len,
        mode=BasicTSMode.TRAIN,
        use_timestamps=True,
        local=True,
        data_file_path="datasets/METR-LA",
        memmap=False,
    )
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    batch = next(iter(loader))

    model = STAEformerGraph(cfg.model_config)
    model.train()

    scaler = Builder._build_scaler(cfg)
    runner = SimpleNamespace(cfg=cfg, scaler=scaler)

    raw_batch = {
        "inputs": batch["inputs"].float(),
        "inputs_timestamps": batch["inputs_timestamps"].float(),
        "targets": batch["targets"].float(),
        "targets_timestamps": batch["targets_timestamps"].float(),
    }
    processed = cfg.taskflow.preprocess(runner, raw_batch)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=cfg.optimizer_params["lr"],
        weight_decay=cfg.optimizer_params["weight_decay"],
    )
    optimizer.zero_grad()

    prediction = model(
        inputs=processed["inputs"],
        inputs_timestamps=processed["inputs_timestamps"],
    )
    loss = cfg.loss(
        prediction=prediction,
        targets=processed["targets"],
        targets_mask=processed["targets_mask"],
        scaler_mean=processed.get("scaler_mean"),
        scaler_std=processed.get("scaler_std"),
    )
    loss.backward()
    optimizer.step()

    graph_bias_grad = None
    if model.graph_bias_table is not None and model.graph_bias_table.grad is not None:
        graph_bias_grad = float(model.graph_bias_table.grad.abs().mean().item())

    return {
        "dataset_batch_shapes": {
            "inputs": tuple(batch["inputs"].shape),
            "inputs_timestamps": tuple(batch["inputs_timestamps"].shape),
            "targets": tuple(batch["targets"].shape),
            "targets_timestamps": tuple(batch["targets_timestamps"].shape),
        },
        "prediction_shape": tuple(prediction.shape),
        "loss": float(loss.item()),
        "graph_bucket_shape": tuple(model.graph_bucket_matrix.shape),
        "graph_bucket_max": int(model.graph_bucket_matrix.max().item()),
        "graph_bias_grad_mean": graph_bias_grad,
    }
