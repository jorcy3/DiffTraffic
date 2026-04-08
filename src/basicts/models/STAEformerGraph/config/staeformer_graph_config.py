from dataclasses import dataclass, field
from typing import Optional

from basicts.models.STAEformer.config.staeformer_config import OFFICIAL_STAEFORMER_PRESETS, STAEformerConfig


@dataclass
class STAEformerGraphConfig(STAEformerConfig):
    """
    STAEformer config with optional physical and semantic graph-aware spatial attention biases.
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
    directional_bias_enabled: bool = field(
        default=True,
        metadata={"help": "Whether to keep directed road-reachability bias separate from the undirected hop prior."},
    )
    directional_hop_radius: int = field(
        default=2,
        metadata={"help": "Maximum hop distance bucketed by the directed graph prior."},
    )
    directional_bias_init: float = field(
        default=0.10,
        metadata={"help": "Initial one-hop bias magnitude for each directed graph prior."},
    )
    semantic_bias_enabled: bool = field(
        default=False,
        metadata={"help": "Whether to inject a data-driven semantic-neighbor bias into spatial attention."},
    )
    semantic_data_file_path: Optional[str] = field(
        default=None,
        metadata={"help": "Dataset directory containing train_data.npy used to build semantic neighbors."},
    )
    semantic_topk: int = field(
        default=8,
        metadata={"help": "Top-k semantic neighbors per node based on training-series similarity."},
    )
    semantic_bias_init: float = field(
        default=0.10,
        metadata={"help": "Initial semantic-neighbor attention bias magnitude."},
    )
    semantic_use_abs_corr: bool = field(
        default=True,
        metadata={"help": "Whether to rank semantic neighbors by absolute Pearson correlation."},
    )


__all__ = ["OFFICIAL_STAEFORMER_PRESETS", "STAEformerGraphConfig"]
