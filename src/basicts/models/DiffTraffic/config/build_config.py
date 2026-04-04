from torch.optim.lr_scheduler import MultiStepLR

from basicts.configs import BasicTSForecastingConfig
from basicts.runners.callback import EarlyStopping
from basicts.models.STAEformer import OFFICIAL_STAEFORMER_PRESETS
from basicts.models.STAEformer.taskflow import STAEformerForecastingTaskFlow
from basicts.models.STAEformer.utils import load_staeformer_scaler_stats

from ..loss import difftraffic_loss
from .difftrafficv1_config import DiffTrafficV1Config


def _resolve_num_features(dataset_name: str) -> int:
    if dataset_name == "METR-LA":
        return 207
    if dataset_name == "PEMS-BAY":
        return 325
    raise ValueError(f"Unsupported dataset_name: {dataset_name}")


def _build_difftraffic_v1_model_config(
    dataset_name: str,
    enable_graph_condition: bool = False,
    residual_hidden_size: int = 128,
    condition_hidden_size: int = 64,
) -> DiffTrafficV1Config:
    return DiffTrafficV1Config(
        input_len=12,
        output_len=12,
        num_features=_resolve_num_features(dataset_name),
        residual_hidden_size=residual_hidden_size,
        condition_hidden_size=condition_hidden_size,
        enable_graph_condition=enable_graph_condition,
        enable_residual_prior=True,
        loss_weight_pred=1.0,
        loss_weight_residual=0.5,
    )


def _build_official_difftraffic_config(
    *,
    model,
    dataset_name: str,
    data_file_path: str,
    gpus: str | None = None,
    batch_size: int | None = None,
    num_epochs: int | None = None,
    num_steps: int | None = None,
    ckpt_save_dir: str | None = None,
    enable_graph_condition: bool = False,
    residual_hidden_size: int = 128,
    condition_hidden_size: int = 64,
) -> BasicTSForecastingConfig:
    if dataset_name not in OFFICIAL_STAEFORMER_PRESETS:
        raise ValueError(f"Unsupported dataset_name: {dataset_name}")

    preset = OFFICIAL_STAEFORMER_PRESETS[dataset_name]
    resolved_batch_size = preset["batch_size"] if batch_size is None else batch_size
    resolved_num_epochs = preset["max_epochs"] if num_epochs is None and num_steps is None else num_epochs

    return BasicTSForecastingConfig(
        model=model,
        model_config=_build_difftraffic_v1_model_config(
            dataset_name=dataset_name,
            enable_graph_condition=enable_graph_condition,
            residual_hidden_size=residual_hidden_size,
            condition_hidden_size=condition_hidden_size,
        ),
        dataset_name=dataset_name,
        input_len=12,
        output_len=12,
        use_timestamps=True,
        gpus=gpus,
        taskflow=STAEformerForecastingTaskFlow(),
        callbacks=[EarlyStopping(preset["early_stop"])],
        loss=difftraffic_loss,
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


def build_official_difftraffic_v1_forecasting_config(
    dataset_name: str,
    data_file_path: str,
    gpus: str | None = None,
    batch_size: int | None = None,
    num_epochs: int | None = None,
    num_steps: int | None = None,
    ckpt_save_dir: str | None = None,
    enable_graph_condition: bool = False,
    residual_hidden_size: int = 128,
    condition_hidden_size: int = 64,
) -> BasicTSForecastingConfig:
    """
    Build an official-semantics training config for DiffTraffic-v1.
    """

    from ..arch.difftrafficv1_arch import DiffTrafficV1ForForecasting

    return _build_official_difftraffic_config(
        model=DiffTrafficV1ForForecasting,
        dataset_name=dataset_name,
        data_file_path=data_file_path,
        gpus=gpus,
        batch_size=batch_size,
        num_epochs=num_epochs,
        num_steps=num_steps,
        ckpt_save_dir=ckpt_save_dir,
        enable_graph_condition=enable_graph_condition,
        residual_hidden_size=residual_hidden_size,
        condition_hidden_size=condition_hidden_size,
    )


def build_official_difftraffic_naive_forecasting_config(
    dataset_name: str,
    data_file_path: str,
    gpus: str | None = None,
    batch_size: int | None = None,
    num_epochs: int | None = None,
    num_steps: int | None = None,
    ckpt_save_dir: str | None = None,
    residual_hidden_size: int = 128,
) -> BasicTSForecastingConfig:
    """
    Build an official-semantics training config for the naive residual ablation.
    """

    from ..arch.difftrafficv1_naive_arch import DiffTrafficV1NaiveForForecasting

    return _build_official_difftraffic_config(
        model=DiffTrafficV1NaiveForForecasting,
        dataset_name=dataset_name,
        data_file_path=data_file_path,
        gpus=gpus,
        batch_size=batch_size,
        num_epochs=num_epochs,
        num_steps=num_steps,
        ckpt_save_dir=ckpt_save_dir,
        enable_graph_condition=False,
        residual_hidden_size=residual_hidden_size,
        condition_hidden_size=64,
    )
