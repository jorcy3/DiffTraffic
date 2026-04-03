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
from basicts.metrics import masked_mae
from basicts.models.STAEformer import (
    OFFICIAL_STAEFORMER_PRESETS,
    STAEformer,
    STAEformerConfig,
    STAEformerForecastingTaskFlow,
    build_official_staeformer_forecasting_config,
    staeformer_official_loss,
)
from basicts.utils.constants import BasicTSMode


def test_staeformer_minimal_forward():
    cfg = STAEformerConfig(input_len=12, output_len=12, num_features=207)
    model = STAEformer(cfg)
    inputs = torch.randn(2, 12, 207)
    inputs_timestamps = torch.zeros(2, 12, 2)
    out = model(inputs, inputs_timestamps)
    assert tuple(out.shape) == (2, 12, 207)


def test_staeformer_official_loss_inverse_transforms_prediction():
    prediction = torch.tensor([[[1.0]]], dtype=torch.float32)
    targets = torch.tensor([[[12.0]]], dtype=torch.float32)
    scaler_mean = torch.tensor(10.0, dtype=torch.float32)
    scaler_std = torch.tensor(2.0, dtype=torch.float32)

    loss = staeformer_official_loss(
        prediction=prediction,
        targets=targets,
        scaler_mean=scaler_mean,
        scaler_std=scaler_std,
    )

    assert torch.allclose(loss, torch.tensor(0.0))


def test_staeformer_real_metrla_single_step():
    dataset_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../datasets/METR-LA"))
    if not os.path.isdir(dataset_path):
        pytest.skip("METR-LA dataset is not available locally.")

    dataset = BasicTSForecastingDataset(
        dataset_name="METR-LA",
        input_len=12,
        output_len=12,
        mode=BasicTSMode.TRAIN,
        use_timestamps=True,
        local=True,
        data_file_path=dataset_path,
        memmap=False,
    )
    batch = next(iter(DataLoader(dataset, batch_size=2, shuffle=False)))

    cfg = STAEformerConfig(input_len=12, output_len=12, num_features=207)
    model = STAEformer(cfg)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)

    opt.zero_grad()
    pred = model(batch["inputs"].float(), batch["inputs_timestamps"].float())
    loss = masked_mae(pred, batch["targets"].float())
    loss.backward()
    opt.step()

    assert tuple(pred.shape) == (2, 12, 207)
    assert torch.isfinite(loss)


def test_staeformer_basicts_metrla_one_step_official_semantics():
    dataset_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../datasets/METR-LA"))
    if not os.path.isdir(dataset_path):
        pytest.skip("METR-LA dataset is not available locally.")

    ckpt_dir = tempfile.mkdtemp(prefix="staeformer_metrla_smoke_")

    try:
        cfg = build_official_staeformer_forecasting_config(
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
        BasicTSLauncher.launch_training(
            cfg
        )
    finally:
        shutil.rmtree(ckpt_dir, ignore_errors=True)


def test_staeformer_build_config_helper_matches_official_semantics():
    dataset_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../datasets/METR-LA"))
    if not os.path.isdir(dataset_path):
        pytest.skip("METR-LA dataset is not available locally.")

    cfg = build_official_staeformer_forecasting_config(
        dataset_name="METR-LA",
        data_file_path=dataset_path,
        gpus=None,
        batch_size=2,
        num_epochs=None,
        num_steps=1,
    )

    assert isinstance(cfg.taskflow, STAEformerForecastingTaskFlow)
    assert cfg.loss is staeformer_official_loss
    assert cfg.norm_each_channel is OFFICIAL_STAEFORMER_PRESETS["METR-LA"]["norm_each_channel"]
    assert cfg.callbacks[0].patience == OFFICIAL_STAEFORMER_PRESETS["METR-LA"]["early_stop"]
    assert torch.is_tensor(cfg.stats["mean"])
    assert torch.is_tensor(cfg.stats["std"])


def test_staeformer_real_pemsbay_single_step():
    dataset_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../datasets/PEMS-BAY"))
    if not os.path.isdir(dataset_path):
        pytest.skip("PEMS-BAY dataset is not available locally.")

    dataset = BasicTSForecastingDataset(
        dataset_name="PEMS-BAY",
        input_len=12,
        output_len=12,
        mode=BasicTSMode.TRAIN,
        use_timestamps=True,
        local=True,
        data_file_path=dataset_path,
        memmap=False,
    )
    batch = next(iter(DataLoader(dataset, batch_size=2, shuffle=False)))

    cfg = STAEformerConfig(input_len=12, output_len=12, num_features=325)
    model = STAEformer(cfg)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)

    opt.zero_grad()
    pred = model(batch["inputs"].float(), batch["inputs_timestamps"].float())
    loss = masked_mae(pred, batch["targets"].float())
    loss.backward()
    opt.step()

    assert tuple(pred.shape) == (2, 12, 325)
    assert torch.isfinite(loss)


def test_staeformer_basicts_pemsbay_one_step_official_semantics():
    dataset_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../datasets/PEMS-BAY"))
    if not os.path.isdir(dataset_path):
        pytest.skip("PEMS-BAY dataset is not available locally.")

    ckpt_dir = tempfile.mkdtemp(prefix="staeformer_pemsbay_smoke_")

    try:
        cfg = build_official_staeformer_forecasting_config(
            dataset_name="PEMS-BAY",
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
        BasicTSLauncher.launch_training(
            cfg
        )
    finally:
        shutil.rmtree(ckpt_dir, ignore_errors=True)
