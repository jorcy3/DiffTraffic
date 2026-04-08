from torch.optim.lr_scheduler import MultiStepLR

from basicts.configs import BasicTSForecastingConfig
from basicts.models.STAEformer import OFFICIAL_STAEFORMER_PRESETS
from basicts.models.STAEformer.loss import staeformer_official_loss
from basicts.models.STAEformer.taskflow import STAEformerForecastingTaskFlow
from basicts.models.STAEformer.utils import load_staeformer_scaler_stats
from basicts.runners.callback import EarlyStopping

from .staeformer_graph_config import STAEformerGraphConfig


def build_official_staeformer_graph_forecasting_config(
    dataset_name: str,
    data_file_path: str,
    gpus: str | None = None,
    batch_size: int | None = None,
    num_epochs: int | None = None,
    num_steps: int | None = None,
    ckpt_save_dir: str | None = None,
    graph_hop_radius: int = 2,
    graph_bias_init: float = 0.20,
    graph_far_bias: float = 0.0,
    directional_bias_enabled: bool = True,
    directional_hop_radius: int = 2,
    directional_bias_init: float = 0.10,
    semantic_bias_enabled: bool = False,
    semantic_topk: int = 8,
    semantic_bias_init: float = 0.10,
    semantic_use_abs_corr: bool = True,
) -> BasicTSForecastingConfig:
    """
    Build an official-style BasicTS config for STAEformerGraph.
    """

    if dataset_name not in OFFICIAL_STAEFORMER_PRESETS:
        raise ValueError(f"Unsupported dataset_name: {dataset_name}")

    from ..arch import STAEformerGraph

    preset = OFFICIAL_STAEFORMER_PRESETS[dataset_name]
    num_features = 207 if dataset_name == "METR-LA" else 325 if dataset_name == "PEMS-BAY" else None
    if num_features is None:
        raise ValueError(f"Unsupported dataset_name: {dataset_name}")

    resolved_batch_size = preset["batch_size"] if batch_size is None else batch_size
    resolved_num_epochs = preset["max_epochs"] if num_epochs is None and num_steps is None else num_epochs

    return BasicTSForecastingConfig(
        model=STAEformerGraph,
        model_config=STAEformerGraphConfig(
            input_len=12,
            output_len=12,
            num_features=num_features,
            graph_bias_enabled=True,
            graph_data_file_path=data_file_path,
            graph_hop_radius=graph_hop_radius,
            graph_bias_init=graph_bias_init,
            graph_far_bias=graph_far_bias,
            directional_bias_enabled=directional_bias_enabled,
            directional_hop_radius=directional_hop_radius,
            directional_bias_init=directional_bias_init,
            semantic_bias_enabled=semantic_bias_enabled,
            semantic_data_file_path=data_file_path,
            semantic_topk=semantic_topk,
            semantic_bias_init=semantic_bias_init,
            semantic_use_abs_corr=semantic_use_abs_corr,
        ),
        dataset_name=dataset_name,
        input_len=12,
        output_len=12,
        use_timestamps=True,
        gpus=gpus,
        taskflow=STAEformerForecastingTaskFlow(),
        callbacks=[EarlyStopping(preset["early_stop"])],
        loss=staeformer_official_loss,
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
