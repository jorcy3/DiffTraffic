from torch.optim.lr_scheduler import MultiStepLR

from basicts.configs import BasicTSForecastingConfig
from basicts.runners.callback import EarlyStopping
from basicts.models.STAEformer import OFFICIAL_STAEFORMER_PRESETS
from basicts.models.STAEformer.utils import load_staeformer_scaler_stats

from ..loss import stmae_pretrain_loss, stmae_style_loss
from ..taskflow import STMAEForecastingTaskFlow, STMAEPretrainTaskFlow
from .stmae_config import STMAEConfig


def build_official_stmae_forecasting_config(
    dataset_name: str,
    data_file_path: str,
    gpus: str | None = None,
    batch_size: int | None = None,
    num_epochs: int | None = None,
    num_steps: int | None = None,
    ckpt_save_dir: str | None = None,
    pretrained_enhancer_ckpt: str | None = None,
) -> BasicTSForecastingConfig:
    """
    Build a BasicTS forecasting config for STAEformer with STMAE-style enhancement.
    """

    if dataset_name not in OFFICIAL_STAEFORMER_PRESETS:
        raise ValueError(f"Unsupported dataset_name: {dataset_name}")

    from ..arch import STMAEForForecasting

    preset = OFFICIAL_STAEFORMER_PRESETS[dataset_name]
    num_features = 207 if dataset_name == "METR-LA" else 325 if dataset_name == "PEMS-BAY" else None
    if num_features is None:
        raise ValueError(f"Unsupported dataset_name: {dataset_name}")

    resolved_batch_size = preset["batch_size"] if batch_size is None else batch_size
    resolved_num_epochs = preset["max_epochs"] if num_epochs is None and num_steps is None else num_epochs

    return BasicTSForecastingConfig(
        model=STMAEForForecasting,
        model_config=STMAEConfig(
            input_len=12,
            output_len=12,
            num_features=num_features,
            pretrained_enhancer_ckpt=pretrained_enhancer_ckpt,
        ),
        dataset_name=dataset_name,
        input_len=12,
        output_len=12,
        use_timestamps=True,
        gpus=gpus,
        taskflow=STMAEForecastingTaskFlow(),
        callbacks=[EarlyStopping(preset["early_stop"])],
        loss=stmae_style_loss,
        metrics=preset["metrics"],
        target_metric="MAE",
        dataset_params={
            "dataset_name": dataset_name,
            "input_len": 12,
            "output_len": 12,
            "use_timestamps": True,
            "memmap": False,
            "local": True,
            "data_file_path": data_file_path,
        },
        null_val=preset["null_val"],
        norm_each_channel=preset["norm_each_channel"],
        stats=load_staeformer_scaler_stats(data_file_path),
        batch_size=resolved_batch_size,
        num_epochs=resolved_num_epochs,
        num_steps=num_steps,
        eval_after_train=True if resolved_num_epochs is not None else False,
        optimizer_params={
            "lr": preset["lr"],
            "weight_decay": preset["weight_decay"],
        },
        lr_scheduler=MultiStepLR,
        lr_scheduler_params={
            "milestones": preset["milestones"],
            "gamma": 0.1,
        },
        ckpt_save_dir=ckpt_save_dir,
    )


def build_official_stmae_pretrain_config(
    dataset_name: str,
    data_file_path: str,
    gpus: str | None = None,
    batch_size: int | None = None,
    num_epochs: int | None = None,
    num_steps: int | None = None,
    ckpt_save_dir: str | None = None,
) -> BasicTSForecastingConfig:
    """
    Build a BasicTS config for the masked pretraining stage of STMAE-style enhancement.
    """

    if dataset_name not in OFFICIAL_STAEFORMER_PRESETS:
        raise ValueError(f"Unsupported dataset_name: {dataset_name}")

    from ..arch import STMAEPretrainer

    preset = OFFICIAL_STAEFORMER_PRESETS[dataset_name]
    num_features = 207 if dataset_name == "METR-LA" else 325 if dataset_name == "PEMS-BAY" else None
    if num_features is None:
        raise ValueError(f"Unsupported dataset_name: {dataset_name}")

    resolved_batch_size = preset["batch_size"] if batch_size is None else batch_size
    resolved_num_epochs = preset["max_epochs"] if num_epochs is None and num_steps is None else num_epochs

    return BasicTSForecastingConfig(
        model=STMAEPretrainer,
        model_config=STMAEConfig(
            input_len=12,
            output_len=12,
            num_features=num_features,
        ),
        dataset_name=dataset_name,
        input_len=12,
        output_len=12,
        use_timestamps=True,
        gpus=gpus,
        taskflow=STMAEPretrainTaskFlow(),
        callbacks=[EarlyStopping(preset["early_stop"])],
        loss=stmae_pretrain_loss,
        metrics=[],
        target_metric="loss",
        dataset_params={
            "dataset_name": dataset_name,
            "input_len": 12,
            "output_len": 12,
            "use_timestamps": True,
            "memmap": False,
            "local": True,
            "data_file_path": data_file_path,
        },
        null_val=preset["null_val"],
        norm_each_channel=preset["norm_each_channel"],
        stats=load_staeformer_scaler_stats(data_file_path),
        batch_size=resolved_batch_size,
        num_epochs=resolved_num_epochs,
        num_steps=num_steps,
        eval_after_train=False,
        optimizer_params={
            "lr": preset["lr"],
            "weight_decay": preset["weight_decay"],
        },
        lr_scheduler=MultiStepLR,
        lr_scheduler_params={
            "milestones": preset["milestones"],
            "gamma": 0.1,
        },
        ckpt_save_dir=ckpt_save_dir,
    )
