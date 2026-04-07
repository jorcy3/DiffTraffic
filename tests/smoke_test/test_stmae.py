# pylint: disable=wrong-import-position
import os
import shutil
import sys
import tempfile

import pytest
import torch
from torch.utils.data import DataLoader

sys.path.append(os.path.abspath(__file__ + "/../../../src/"))
os.chdir(os.path.abspath(os.path.join(os.path.dirname(__file__))))

from basicts import BasicTSLauncher
from basicts.data import BasicTSForecastingDataset
from basicts.models.STMAE import (
    STMAEConfig,
    STMAEForForecasting,
    STMAEForecastingTaskFlow,
    STMAEPretrainTaskFlow,
    STMAEPretrainer,
    build_official_stmae_forecasting_config,
    build_official_stmae_pretrain_config,
    stmae_pretrain_loss,
    stmae_style_loss,
)
from basicts.utils.constants import BasicTSMode


def test_stmae_minimal_forward():
    cfg = STMAEConfig(input_len=12, output_len=12, num_features=207)
    model = STMAEForForecasting(cfg)
    inputs = torch.randn(2, 12, 207)
    inputs_timestamps = torch.zeros(2, 12, 2)
    outputs = model(inputs, inputs_timestamps)
    assert tuple(outputs["prediction"].shape) == (2, 12, 207)
    assert tuple(outputs["masked_reconstruction"].shape) == (2, 12, 207)


def test_stmae_pretrainer_minimal_forward():
    cfg = STMAEConfig(input_len=12, output_len=12, num_features=207)
    model = STMAEPretrainer(cfg)
    outputs = model(torch.randn(2, 12, 207), torch.zeros(2, 12, 2))
    assert tuple(outputs["prediction"].shape) == (2, 12, 207)
    assert tuple(outputs["masked_positions"].shape) == (2, 12, 207)


def test_stmae_pretrainer_eval_keeps_masking_for_validation():
    cfg = STMAEConfig(
        input_len=12,
        output_len=12,
        num_features=207,
        spatial_mask_ratio=1.0,
        temporal_mask_ratio=0.0,
    )
    model = STMAEPretrainer(cfg)
    model.eval()
    outputs = model(torch.randn(2, 12, 207), torch.zeros(2, 12, 2))
    assert torch.any(outputs["masked_positions"])


def test_stmae_loss_combines_forecast_and_reconstruction():
    prediction = torch.tensor([[[1.0]]], dtype=torch.float32)
    targets = torch.tensor([[[12.0]]], dtype=torch.float32)
    masked_reconstruction = torch.tensor([[[2.0]]], dtype=torch.float32)
    enhancement_targets = torch.tensor([[[3.0]]], dtype=torch.float32)
    masked_positions = torch.tensor([[[True]]])
    scaler_mean = torch.tensor(10.0, dtype=torch.float32)
    scaler_std = torch.tensor(2.0, dtype=torch.float32)
    aux_info = {"loss_weights": {"prediction": 1.0, "reconstruction": 0.5}}

    loss = stmae_style_loss(
        prediction=prediction,
        targets=targets,
        masked_reconstruction=masked_reconstruction,
        masked_positions=masked_positions,
        enhancement_targets=enhancement_targets,
        scaler_mean=scaler_mean,
        scaler_std=scaler_std,
        aux_info=aux_info,
    )

    assert torch.allclose(loss, torch.tensor(0.5))


def _run_real_single_step(dataset_name: str, num_features: int):
    dataset_path = os.path.abspath(os.path.join(os.path.dirname(__file__), f"../../datasets/{dataset_name}"))
    if not os.path.isdir(dataset_path):
        pytest.skip(f"{dataset_name} dataset is not available locally.")

    dataset = BasicTSForecastingDataset(
        dataset_name=dataset_name,
        input_len=12,
        output_len=12,
        mode=BasicTSMode.TRAIN,
        use_timestamps=True,
        local=True,
        data_file_path=dataset_path,
        memmap=False,
    )
    batch = next(iter(DataLoader(dataset, batch_size=2, shuffle=False)))

    cfg = STMAEConfig(input_len=12, output_len=12, num_features=num_features)
    model = STMAEForForecasting(cfg)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    optimizer.zero_grad()
    outputs = model(batch["inputs"].float(), batch["inputs_timestamps"].float())
    loss = stmae_style_loss(
        prediction=outputs["prediction"],
        targets=batch["targets"].float(),
        masked_reconstruction=outputs["masked_reconstruction"],
        masked_positions=outputs["masked_positions"],
        enhancement_targets=batch["inputs"].float(),
        enhancement_valid_mask=torch.ones_like(batch["inputs"], dtype=torch.bool),
        aux_info=outputs.get("aux_info"),
    )
    loss.backward()
    optimizer.step()

    assert tuple(outputs["prediction"].shape) == (2, 12, num_features)
    assert torch.isfinite(loss)


def test_stmae_real_metrla_single_step():
    _run_real_single_step("METR-LA", 207)


def test_stmae_real_pemsbay_single_step():
    _run_real_single_step("PEMS-BAY", 325)


def test_stmae_build_config_helper_matches_mainline_semantics():
    dataset_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../datasets/METR-LA"))
    if not os.path.isdir(dataset_path):
        pytest.skip("METR-LA dataset is not available locally.")

    cfg = build_official_stmae_forecasting_config(
        dataset_name="METR-LA",
        data_file_path=dataset_path,
        gpus=None,
        batch_size=2,
        num_epochs=None,
        num_steps=1,
    )

    assert isinstance(cfg.taskflow, STMAEForecastingTaskFlow)
    assert cfg.loss is stmae_style_loss


def test_stmae_pretrain_build_config_helper_matches_mainline_semantics():
    dataset_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../datasets/METR-LA"))
    if not os.path.isdir(dataset_path):
        pytest.skip("METR-LA dataset is not available locally.")

    cfg = build_official_stmae_pretrain_config(
        dataset_name="METR-LA",
        data_file_path=dataset_path,
        gpus=None,
        batch_size=2,
        num_epochs=None,
        num_steps=1,
    )

    assert isinstance(cfg.taskflow, STMAEPretrainTaskFlow)
    assert cfg.loss is stmae_pretrain_loss
    assert cfg.target_metric == "loss"


def test_stmae_basicts_metrla_one_step():
    dataset_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../datasets/METR-LA"))
    if not os.path.isdir(dataset_path):
        pytest.skip("METR-LA dataset is not available locally.")

    ckpt_dir = tempfile.mkdtemp(prefix="stmae_metrla_smoke_")
    try:
        cfg = build_official_stmae_forecasting_config(
            dataset_name="METR-LA",
            data_file_path=dataset_path,
            gpus=None,
            batch_size=2,
            num_epochs=None,
            num_steps=1,
            ckpt_save_dir=ckpt_dir,
        )
        cfg.metrics = ["MAE"]
        cfg.target_metric = "MAE"
        cfg.val_interval = 999999
        cfg.test_interval = 999999
        cfg.eval_after_train = False
        BasicTSLauncher.launch_training(cfg)
    finally:
        shutil.rmtree(ckpt_dir, ignore_errors=True)


def test_stmae_pretrain_basicts_metrla_one_step():
    dataset_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../datasets/METR-LA"))
    if not os.path.isdir(dataset_path):
        pytest.skip("METR-LA dataset is not available locally.")

    ckpt_dir = tempfile.mkdtemp(prefix="stmae_pretrain_metrla_smoke_")
    try:
        cfg = build_official_stmae_pretrain_config(
            dataset_name="METR-LA",
            data_file_path=dataset_path,
            gpus=None,
            batch_size=2,
            num_epochs=None,
            num_steps=1,
            ckpt_save_dir=ckpt_dir,
        )
        cfg.val_interval = 999999
        cfg.test_interval = 999999
        cfg.eval_after_train = False
        BasicTSLauncher.launch_training(cfg)
    finally:
        shutil.rmtree(ckpt_dir, ignore_errors=True)
