import torch
from torch import nn

from ..config.staeformer_config import STAEformerConfig


class AttentionLayer(nn.Module):
    """
    Perform attention across the second-to-last dimension.
    """

    def __init__(self, model_dim: int, num_heads: int = 8, mask: bool = False):
        super().__init__()
        if model_dim % num_heads != 0:
            raise ValueError(f"model_dim ({model_dim}) must be divisible by num_heads ({num_heads}).")

        self.model_dim = model_dim
        self.num_heads = num_heads
        self.mask = mask
        self.head_dim = model_dim // num_heads

        self.fc_q = nn.Linear(model_dim, model_dim)
        self.fc_k = nn.Linear(model_dim, model_dim)
        self.fc_v = nn.Linear(model_dim, model_dim)
        self.out_proj = nn.Linear(model_dim, model_dim)

    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor) -> torch.Tensor:
        """
        Shape:
            query: [B, ..., Lt, D]
            key: [B, ..., Ls, D]
            value: [B, ..., Ls, D]
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

        attn_score = torch.softmax(attn_score, dim=-1)
        out = attn_score @ value
        out = torch.cat(torch.split(out, batch_size, dim=0), dim=-1)
        return self.out_proj(out)


class SelfAttentionLayer(nn.Module):
    """
    Vanilla transformer self-attention block used by STAEformer.
    """

    def __init__(
        self,
        model_dim: int,
        feed_forward_dim: int = 2048,
        num_heads: int = 8,
        dropout: float = 0.0,
        mask: bool = False,
    ):
        super().__init__()
        self.attn = AttentionLayer(model_dim, num_heads, mask)
        self.feed_forward = nn.Sequential(
            nn.Linear(model_dim, feed_forward_dim),
            nn.ReLU(inplace=True),
            nn.Linear(feed_forward_dim, model_dim),
        )
        self.ln1 = nn.LayerNorm(model_dim)
        self.ln2 = nn.LayerNorm(model_dim)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, dim: int = -2) -> torch.Tensor:
        """
        Shape:
            x: [B, ..., L, D]
            return: [B, ..., L, D]
        """

        x = x.transpose(dim, -2)
        residual = x
        out = self.attn(x, x, x)
        out = self.dropout1(out)
        out = self.ln1(residual + out)

        residual = out
        out = self.feed_forward(out)
        out = self.dropout2(out)
        out = self.ln2(residual + out)
        return out.transpose(dim, -2)


class STAEformer(nn.Module):
    """
    BasicTS-native STAEformer adapted from the official implementation.

    Official input:
        x: [B, I, N, 3] where channels are [value, tod, dow]

    BasicTS input:
        inputs: [B, I, N]
        inputs_timestamps: [B, I, 2] where channels are [tod_norm, dow_norm]
    """

    def __init__(self, config: STAEformerConfig):
        super().__init__()

        self.num_nodes = config.num_nodes if config.num_nodes is not None else config.num_features
        self.in_steps = config.input_len
        self.out_steps = config.output_len
        self.steps_per_day = config.steps_per_day
        self.num_day_in_week = config.num_day_in_week
        self.input_dim = config.input_dim
        self.output_dim = config.output_dim
        self.input_embedding_dim = config.input_embedding_dim
        self.tod_embedding_dim = config.tod_embedding_dim
        self.dow_embedding_dim = config.dow_embedding_dim
        self.spatial_embedding_dim = config.spatial_embedding_dim
        self.adaptive_embedding_dim = config.adaptive_embedding_dim
        self.model_dim = (
            self.input_embedding_dim
            + self.tod_embedding_dim
            + self.dow_embedding_dim
            + self.spatial_embedding_dim
            + self.adaptive_embedding_dim
        )
        self.num_heads = config.num_heads
        self.num_layers = config.num_layers
        self.use_mixed_proj = config.use_mixed_proj

        if self.input_dim != 3:
            raise ValueError("BasicTS-native STAEformer requires input_dim=3 to match the official formulation.")
        if self.output_dim != 1:
            raise ValueError("BasicTS-native STAEformer currently requires output_dim=1 to match forecasting targets.")

        self.input_proj = nn.Linear(self.input_dim, self.input_embedding_dim)
        if self.tod_embedding_dim > 0:
            self.tod_embedding = nn.Embedding(self.steps_per_day, self.tod_embedding_dim)
        if self.dow_embedding_dim > 0:
            self.dow_embedding = nn.Embedding(self.num_day_in_week, self.dow_embedding_dim)
        if self.spatial_embedding_dim > 0:
            self.node_emb = nn.Parameter(torch.empty(self.num_nodes, self.spatial_embedding_dim))
            nn.init.xavier_uniform_(self.node_emb)
        if self.adaptive_embedding_dim > 0:
            self.adaptive_embedding = nn.init.xavier_uniform_(
                nn.Parameter(torch.empty(self.in_steps, self.num_nodes, self.adaptive_embedding_dim))
            )

        if self.use_mixed_proj:
            self.output_proj = nn.Linear(self.in_steps * self.model_dim, self.out_steps * self.output_dim)
        else:
            self.temporal_proj = nn.Linear(self.in_steps, self.out_steps)
            self.output_proj = nn.Linear(self.model_dim, self.output_dim)

        self.attn_layers_t = nn.ModuleList(
            [
                SelfAttentionLayer(
                    self.model_dim,
                    config.feed_forward_dim,
                    self.num_heads,
                    config.dropout,
                )
                for _ in range(self.num_layers)
            ]
        )
        self.attn_layers_s = nn.ModuleList(
            [
                SelfAttentionLayer(
                    self.model_dim,
                    config.feed_forward_dim,
                    self.num_heads,
                    config.dropout,
                )
                for _ in range(self.num_layers)
            ]
        )

    def _recover_time_indices(self, inputs_timestamps: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        tod_index = torch.floor(inputs_timestamps[..., 0] * self.steps_per_day).long()
        tod_index = tod_index.clamp(min=0, max=self.steps_per_day - 1)

        dow_index = torch.floor(inputs_timestamps[..., 1] * self.num_day_in_week + 1e-6).long()
        dow_index = dow_index.clamp(min=0, max=self.num_day_in_week - 1)
        return tod_index, dow_index

    def _build_official_input(
        self,
        inputs: torch.Tensor,
        inputs_timestamps: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
        """
        Shape:
            inputs: [B, I, N]
            inputs_timestamps: [B, I, 2] or None
            return:
                official_input: [B, I, N, 3]
                tod_index: [B, I] or None
                dow_index: [B, I] or None
        """

        if inputs_timestamps is None:
            tod_norm = torch.zeros(inputs.shape[0], inputs.shape[1], device=inputs.device, dtype=inputs.dtype)
            dow_raw = torch.zeros(inputs.shape[0], inputs.shape[1], device=inputs.device, dtype=inputs.dtype)
            tod = None
            dow = None
        else:
            tod, dow = self._recover_time_indices(inputs_timestamps)
            tod_norm = inputs_timestamps[..., 0]
            dow_raw = dow.to(inputs.dtype)

        official_input = torch.stack(
            [
                inputs,
                tod_norm.unsqueeze(-1).expand(-1, -1, self.num_nodes),
                dow_raw.unsqueeze(-1).expand(-1, -1, self.num_nodes),
            ],
            dim=-1,
        )

        return official_input, tod, dow

    def _expand_temporal_indices(self, indices: torch.Tensor, num_nodes: int) -> torch.Tensor:
        """
        Shape:
            indices: [B, I]
            return: [B, I, N]
        """

        return indices.unsqueeze(-1).expand(-1, -1, num_nodes)

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
        for attn in self.attn_layers_s:
            x = attn(x, dim=2)

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
