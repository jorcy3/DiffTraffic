from dataclasses import dataclass, field
from typing import Optional

from basicts.configs import BasicTSModelConfig


OFFICIAL_STAEFORMER_PRESETS = {
    "METR-LA": {
        "lr": 1e-3,
        "weight_decay": 3e-4,
        "milestones": [20, 30],
        "batch_size": 16,
        "max_epochs": 200,
        "early_stop": 30,
        "norm_each_channel": False,
        "null_val": 0.0,
        "metrics": ["MAE", "RMSE", "MAPE"],
    },
    "PEMS-BAY": {
        "lr": 1e-3,
        "weight_decay": 1e-4,
        "milestones": [10, 30],
        "batch_size": 16,
        "max_epochs": 300,
        "early_stop": 20,
        "norm_each_channel": False,
        "null_val": 0.0,
        "metrics": ["MAE", "RMSE", "MAPE"],
    },
}


@dataclass
class STAEformerConfig(BasicTSModelConfig):
    """
    Config class for STAEformer.

    Notes:
        In BasicTS traffic forecasting, `num_features` is the number of sensors/nodes.
        This differs from the official repo, where `num_nodes` is explicit and
        `input_dim/output_dim` describe per-node channels.
    """

    input_len: int = field(default=None, metadata={"help": "Input sequence length."})
    output_len: int = field(default=None, metadata={"help": "Output sequence length."})
    num_features: int = field(default=None, metadata={"help": "Number of traffic sensors/nodes."})
    num_nodes: Optional[int] = field(
        default=None,
        metadata={"help": "Optional explicit node count. If None, use num_features."},
    )

    steps_per_day: int = field(default=288, metadata={"help": "Time steps per day."})
    num_day_in_week: int = field(default=7, metadata={"help": "Days in a week."})

    input_dim: int = field(
        default=3,
        metadata={"help": "Per-node input channel count in the official formulation."},
    )
    output_dim: int = field(
        default=1,
        metadata={"help": "Per-node output channel count in the official formulation."},
    )
    input_embedding_dim: int = field(default=24, metadata={"help": "Value embedding dimension."})
    tod_embedding_dim: int = field(default=24, metadata={"help": "Time-of-day embedding dimension."})
    dow_embedding_dim: int = field(default=24, metadata={"help": "Day-of-week embedding dimension."})
    spatial_embedding_dim: int = field(default=0, metadata={"help": "Spatial embedding dimension."})
    adaptive_embedding_dim: int = field(default=80, metadata={"help": "Adaptive embedding dimension."})
    feed_forward_dim: int = field(default=256, metadata={"help": "Feed-forward hidden dimension."})
    num_heads: int = field(default=4, metadata={"help": "Attention head count."})
    num_layers: int = field(default=3, metadata={"help": "Number of temporal/spatial attention blocks."})
    dropout: float = field(default=0.1, metadata={"help": "Dropout rate."})
    use_mixed_proj: bool = field(default=True, metadata={"help": "Whether to use the official mixed projection head."})
