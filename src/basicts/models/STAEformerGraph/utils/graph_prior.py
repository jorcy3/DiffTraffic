from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from basicts.utils.serialization import load_pkl


def load_raw_adjacency(data_file_path: str) -> np.ndarray:
    """
    Load the raw adjacency matrix from a BasicTS traffic dataset directory.
    """

    adj_path = Path(data_file_path) / "adj_mx.pkl"
    if not adj_path.exists():
        raise FileNotFoundError(f"adj_mx.pkl not found under dataset directory: {data_file_path}")

    obj = load_pkl(str(adj_path))
    if isinstance(obj, (list, tuple)) and len(obj) >= 3 and hasattr(obj[2], "shape"):
        adjacency = np.asarray(obj[2], dtype=np.float32)
    elif hasattr(obj, "shape"):
        adjacency = np.asarray(obj, dtype=np.float32)
    else:
        raise ValueError(f"Unsupported adjacency file format: {type(obj)}")

    if adjacency.ndim != 2 or adjacency.shape[0] != adjacency.shape[1]:
        raise ValueError(f"Expected a square adjacency matrix, got shape {adjacency.shape}.")

    return adjacency


def load_train_series(data_file_path: str) -> np.ndarray:
    """
    Load the training series matrix used to build semantic neighbors.
    """

    train_path = Path(data_file_path) / "train_data.npy"
    if not train_path.exists():
        raise FileNotFoundError(f"train_data.npy not found under dataset directory: {data_file_path}")

    series = np.load(train_path)
    if series.ndim == 3:
        # Keep the first channel if a multichannel format is provided.
        series = series[..., 0]
    if series.ndim != 2:
        raise ValueError(f"Expected train_data.npy with shape [T, N], got {series.shape}.")
    return np.asarray(series, dtype=np.float32)


def _build_reachability_bucket_matrix(adjacency_bool: np.ndarray, hop_radius: int) -> torch.Tensor:
    """
    Convert a boolean reachability graph into shortest-hop buckets.
    """

    if hop_radius < 1:
        raise ValueError(f"hop_radius must be >= 1, got {hop_radius}.")

    adjacency_bool = np.asarray(adjacency_bool, dtype=bool)
    if adjacency_bool.ndim != 2 or adjacency_bool.shape[0] != adjacency_bool.shape[1]:
        raise ValueError(f"Expected a square boolean adjacency matrix, got shape {adjacency_bool.shape}.")

    adjacency_bool = adjacency_bool.copy()
    np.fill_diagonal(adjacency_bool, False)

    num_nodes = adjacency_bool.shape[0]
    buckets = np.full((num_nodes, num_nodes), hop_radius + 1, dtype=np.int64)
    np.fill_diagonal(buckets, 0)

    visited = np.eye(num_nodes, dtype=bool)
    reach = adjacency_bool.copy()
    adjacency_int = adjacency_bool.astype(np.int8)

    for hop in range(1, hop_radius + 1):
        new_reach = np.logical_and(reach, ~visited)
        buckets[new_reach] = hop
        visited = np.logical_or(visited, new_reach)
        reach = (reach.astype(np.int8) @ adjacency_int) > 0

    return torch.from_numpy(buckets).long()


def build_hop_bucket_matrix(adjacency: np.ndarray, hop_radius: int) -> torch.Tensor:
    """
    Convert an adjacency matrix into shortest-hop buckets capped at `hop_radius + 1`.

    Bucket meaning:
        0: self
        1..hop_radius: nodes within that hop distance
        hop_radius + 1: nodes beyond the hop radius
    """

    adjacency_bool = np.asarray(adjacency > 0, dtype=bool)
    adjacency_bool = np.logical_or(adjacency_bool, adjacency_bool.T)
    return _build_reachability_bucket_matrix(adjacency_bool=adjacency_bool, hop_radius=hop_radius)


def build_directional_hop_bucket_matrix(adjacency: np.ndarray, hop_radius: int) -> torch.Tensor:
    """
    Convert a directed adjacency matrix into shortest-hop buckets without symmetrization.

    Bucket meaning:
        0: self
        1..hop_radius: nodes reachable along the directed graph in that many hops
        hop_radius + 1: nodes not reachable within the hop radius
    """

    adjacency_bool = np.asarray(adjacency > 0, dtype=bool)
    return _build_reachability_bucket_matrix(adjacency_bool=adjacency_bool, hop_radius=hop_radius)


def build_semantic_bucket_matrix(
    train_series: np.ndarray,
    topk: int,
    use_abs_corr: bool = True,
) -> torch.Tensor:
    """
    Build a semantic-neighbor bucket matrix from training-series correlations.

    Bucket meaning:
        0: self
        1: semantic neighbor
        2: otherwise
    """

    if topk < 1:
        raise ValueError(f"semantic topk must be >= 1, got {topk}.")

    if train_series.ndim != 2:
        raise ValueError(f"Expected train_series with shape [T, N], got {train_series.shape}.")

    num_nodes = train_series.shape[1]
    corr = np.corrcoef(train_series, rowvar=False)
    corr = np.nan_to_num(corr, nan=0.0, posinf=0.0, neginf=0.0)
    if use_abs_corr:
        corr = np.abs(corr)

    np.fill_diagonal(corr, -np.inf)
    topk = min(topk, max(num_nodes - 1, 1))

    semantic_mask = np.zeros((num_nodes, num_nodes), dtype=bool)
    topk_indices = np.argpartition(corr, kth=-topk, axis=1)[:, -topk:]
    row_index = np.arange(num_nodes)[:, None]
    semantic_mask[row_index, topk_indices] = True
    semantic_mask = np.logical_or(semantic_mask, semantic_mask.T)

    buckets = np.full((num_nodes, num_nodes), 2, dtype=np.int64)
    np.fill_diagonal(buckets, 0)
    buckets[semantic_mask] = 1
    np.fill_diagonal(buckets, 0)
    return torch.from_numpy(buckets).long()


def resolve_graph_bucket_matrix(
    graph_data_file_path: str | None,
    num_nodes: int,
    hop_radius: int,
) -> torch.Tensor | None:
    """
    Build the hop bucket matrix for the configured traffic graph.
    """

    if graph_data_file_path is None:
        return None

    adjacency = load_raw_adjacency(graph_data_file_path)
    if adjacency.shape[0] != num_nodes:
        raise ValueError(
            f"Graph node count mismatch: adjacency has {adjacency.shape[0]} nodes, config expects {num_nodes}."
        )
    return build_hop_bucket_matrix(adjacency, hop_radius)


def resolve_directional_bucket_matrices(
    graph_data_file_path: str | None,
    num_nodes: int,
    hop_radius: int,
) -> tuple[torch.Tensor | None, torch.Tensor | None]:
    """
    Build forward and backward directed-hop bucket matrices for the configured traffic graph.
    """

    if graph_data_file_path is None:
        return None, None

    adjacency = load_raw_adjacency(graph_data_file_path)
    if adjacency.shape[0] != num_nodes:
        raise ValueError(
            f"Graph node count mismatch: adjacency has {adjacency.shape[0]} nodes, config expects {num_nodes}."
        )

    forward_buckets = build_directional_hop_bucket_matrix(adjacency, hop_radius=hop_radius)
    backward_buckets = build_directional_hop_bucket_matrix(adjacency.T, hop_radius=hop_radius)
    return forward_buckets, backward_buckets


def resolve_semantic_bucket_matrix(
    semantic_data_file_path: str | None,
    num_nodes: int,
    topk: int,
    use_abs_corr: bool = True,
) -> torch.Tensor | None:
    """
    Build the semantic-neighbor bucket matrix from training data.
    """

    if semantic_data_file_path is None:
        return None

    train_series = load_train_series(semantic_data_file_path)
    if train_series.shape[1] != num_nodes:
        raise ValueError(
            f"Semantic series node count mismatch: train_data has {train_series.shape[1]} nodes, config expects {num_nodes}."
        )
    return build_semantic_bucket_matrix(train_series, topk=topk, use_abs_corr=use_abs_corr)
