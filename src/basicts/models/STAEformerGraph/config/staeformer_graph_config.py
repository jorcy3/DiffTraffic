from dataclasses import dataclass, field
from typing import Optional

from basicts.models.STAEformer.config.staeformer_config import OFFICIAL_STAEFORMER_PRESETS, STAEformerConfig


@dataclass
class STAEformerGraphConfig(STAEformerConfig):
    """
    STAEformer config with an optional graph-aware spatial attention bias.
    """

    graph_bias_enabled: bool = field(
        default=True,
        metadata={"help": "Whether to inject graph-aware bias into spatial attention."},
    )
    graph_data_file_path: Optional[str] = field(
        default=None,
        metadata={"help": "Dataset directory containing adj_mx.pkl used to build graph priors."},
    )
    graph_hop_radius: int = field(
        default=2,
        metadata={"help": "Maximum hop distance bucketed by the graph prior."},
    )
    graph_bias_init: float = field(
        default=0.20,
        metadata={"help": "Initial one-hop attention bias magnitude."},
    )
    graph_far_bias: float = field(
        default=0.0,
        metadata={"help": "Initial bias for nodes beyond the hop radius."},
    )


__all__ = ["OFFICIAL_STAEFORMER_PRESETS", "STAEformerGraphConfig"]
