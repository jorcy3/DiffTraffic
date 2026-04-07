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


def build_hop_bucket_matrix(adjacency: np.ndarray, hop_radius: int) -> torch.Tensor:
    """
    Convert an adjacency matrix into shortest-hop buckets capped at `hop_radius + 1`.

    Bucket meaning:
        0: self
        1..hop_radius: nodes within that hop distance
        hop_radius + 1: nodes beyond the hop radius
    """

    if hop_radius < 1:
        raise ValueError(f"hop_radius must be >= 1, got {hop_radius}.")

    adjacency_bool = np.asarray(adjacency > 0, dtype=bool)
    adjacency_bool = np.logical_or(adjacency_bool, adjacency_bool.T)
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
