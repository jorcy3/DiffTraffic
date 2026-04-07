from __future__ import annotations

import torch
from torch import nn

from basicts.models.STAEformer.arch.staeformer_arch import AttentionLayer, STAEformer

from ..config.staeformer_graph_config import STAEformerGraphConfig
from ..utils import resolve_graph_bucket_matrix


class GraphAttentionLayer(AttentionLayer):
    """
    STAEformer attention layer with an optional additive attention bias.
    """

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attn_bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Shape:
            query: [B, ..., Lt, D]
            key: [B, ..., Ls, D]
            value: [B, ..., Ls, D]
            attn_bias: [H, Lt, Ls] or None
            return: [B, ..., Lt, D]
        """

        batch_size = query.shape[0]
        tgt_length = query.shape[-2]
        src_length = key.shape[-2]

        query = self.fc_q(query)
        key = self.fc_k(key)
        value = self.fc_v(value)

        query = torch.cat(torch.split(query, self.head_dim, dim=-1), dim=0)
        key = torch.cat(torch.split(key, self.head_dim, dim=-1), dim=0)
        value = torch.cat(torch.split(value, self.head_dim, dim=-1), dim=0)

        key = key.transpose(-1, -2)
        attn_score = (query @ key) / (self.head_dim**0.5)

        if self.mask:
            causal_mask = torch.ones(
                tgt_length,
                src_length,
                dtype=torch.bool,
                device=query.device,
            ).tril()
            attn_score.masked_fill_(~causal_mask, -torch.inf)

        if attn_bias is not None:
            expanded_bias = attn_bias.repeat(batch_size, 1, 1)
            while expanded_bias.ndim < attn_score.ndim:
                expanded_bias = expanded_bias.unsqueeze(1)
            attn_score = attn_score + expanded_bias

        attn_score = torch.softmax(attn_score, dim=-1)
        out = attn_score @ value
        out = torch.cat(torch.split(out, batch_size, dim=0), dim=-1)
        return self.out_proj(out)


class GraphSelfAttentionLayer(nn.Module):
    """
    STAEformer self-attention block with graph-aware spatial bias support.
    """

    def __init__(
        self,
        model_dim: int,
        feed_forward_dim: int = 2048,
        num_heads: int = 8,
        dropout: float = 0.0,
        mask: bool = False,
    ) -> None:
        super().__init__()
        self.attn = GraphAttentionLayer(model_dim, num_heads, mask)
        self.feed_forward = nn.Sequential(
            nn.Linear(model_dim, feed_forward_dim),
            nn.ReLU(inplace=True),
            nn.Linear(feed_forward_dim, model_dim),
        )
        self.ln1 = nn.LayerNorm(model_dim)
        self.ln2 = nn.LayerNorm(model_dim)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        dim: int = -2,
        attn_bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Shape:
            x: [B, ..., L, D]
            attn_bias: [H, L, L] or None
            return: [B, ..., L, D]
        """

        x = x.transpose(dim, -2)
        residual = x
        out = self.attn(x, x, x, attn_bias=attn_bias)
        out = self.dropout1(out)
        out = self.ln1(residual + out)

        residual = out
        out = self.feed_forward(out)
        out = self.dropout2(out)
        out = self.ln2(residual + out)
        return out.transpose(dim, -2)


class STAEformerGraph(STAEformer):
    """
    STAEformer with PDFormer-inspired graph-aware spatial attention bias.

    This is a thin migration of the traffic graph prior into STAEformer's
    spatial attention, not a full re-implementation of PDFormer.
    """

    def __init__(self, config: STAEformerGraphConfig):
        super().__init__(config)
        self.graph_bias_enabled = bool(config.graph_bias_enabled)
        self.graph_hop_radius = int(config.graph_hop_radius)

        self.attn_layers_s = nn.ModuleList(
            [
                GraphSelfAttentionLayer(
                    self.model_dim,
                    config.feed_forward_dim,
                    self.num_heads,
                    config.dropout,
                )
                for _ in range(self.num_layers)
            ]
        )

        if self.graph_bias_enabled:
            graph_bucket_matrix = resolve_graph_bucket_matrix(
                graph_data_file_path=config.graph_data_file_path,
                num_nodes=self.num_nodes,
                hop_radius=self.graph_hop_radius,
            )
            if graph_bucket_matrix is None:
                raise ValueError("graph_bias_enabled=True requires `graph_data_file_path` to be set.")
            self.register_buffer("graph_bucket_matrix", graph_bucket_matrix, persistent=False)
            self.graph_bias_table = nn.Parameter(self._build_initial_bias_table(config))
        else:
            self.register_buffer("graph_bucket_matrix", torch.empty(0, 0, dtype=torch.long), persistent=False)
            self.graph_bias_table = None

    def _build_initial_bias_table(self, config: STAEformerGraphConfig) -> torch.Tensor:
        """
        Initialize hop-bucket biases with stronger preference for closer nodes.
        """

        num_buckets = self.graph_hop_radius + 2
        base_bias = torch.full((num_buckets,), float(config.graph_far_bias), dtype=torch.float32)
        base_bias[0] = 0.0
        for hop in range(1, self.graph_hop_radius + 1):
            base_bias[hop] = float(config.graph_bias_init) / hop
        return base_bias.unsqueeze(0).repeat(self.num_heads, 1)

    def _build_spatial_attn_bias(self) -> torch.Tensor | None:
        """
        Build per-head additive attention bias from hop buckets.
        """

        if not self.graph_bias_enabled or self.graph_bias_table is None or self.graph_bucket_matrix.numel() == 0:
            return None
        return self.graph_bias_table[:, self.graph_bucket_matrix]

    def forward(
        self,
        inputs: torch.Tensor,
        inputs_timestamps: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Shape:
            inputs: [B, I, N]
            inputs_timestamps: [B, I, 2] or None
            return: [B, O, N]
        """

        if inputs.ndim != 3:
            raise ValueError(f"Expected `inputs` with shape [B, I, N], got {tuple(inputs.shape)}.")
        if inputs.shape[2] != self.num_nodes:
            raise ValueError(
                f"Expected num_nodes={self.num_nodes} from config, but got inputs.shape[2]={inputs.shape[2]}."
            )

        batch_size = inputs.shape[0]
        official_input, tod_index, dow_index = self._build_official_input(inputs, inputs_timestamps)

        x = self.input_proj(official_input)
        features = [x]

        if self.tod_embedding_dim > 0:
            if tod_index is None:
                tod_index = torch.zeros(
                    inputs.shape[0],
                    inputs.shape[1],
                    dtype=torch.long,
                    device=inputs.device,
                )
            tod_index = self._expand_temporal_indices(tod_index, self.num_nodes)
            features.append(self.tod_embedding(tod_index))

        if self.dow_embedding_dim > 0:
            if dow_index is None:
                dow_index = torch.zeros(
                    inputs.shape[0],
                    inputs.shape[1],
                    dtype=torch.long,
                    device=inputs.device,
                )
            dow_index = self._expand_temporal_indices(dow_index, self.num_nodes)
            features.append(self.dow_embedding(dow_index))

        if self.spatial_embedding_dim > 0:
            spatial_emb = self.node_emb.expand(batch_size, self.in_steps, *self.node_emb.shape)
            features.append(spatial_emb)

        if self.adaptive_embedding_dim > 0:
            adaptive_emb = self.adaptive_embedding.expand(size=(batch_size, *self.adaptive_embedding.shape))
            features.append(adaptive_emb)

        x = torch.cat(features, dim=-1)

        for attn in self.attn_layers_t:
            x = attn(x, dim=1)

        spatial_attn_bias = self._build_spatial_attn_bias()
        for attn in self.attn_layers_s:
            x = attn(x, dim=2, attn_bias=spatial_attn_bias)

        if self.use_mixed_proj:
            out = x.transpose(1, 2)
            out = out.reshape(batch_size, self.num_nodes, self.in_steps * self.model_dim)
            out = self.output_proj(out).view(batch_size, self.num_nodes, self.out_steps, self.output_dim)
            out = out.transpose(1, 2)
        else:
            out = x.transpose(1, 3)
            out = self.temporal_proj(out)
            out = self.output_proj(out.transpose(1, 3))

        return out.squeeze(-1)
