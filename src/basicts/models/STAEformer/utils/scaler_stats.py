import os

import numpy as np
import torch


def load_staeformer_scaler_stats(data_file_path: str) -> dict[str, torch.Tensor]:
    """
    Load official-style global scaler statistics from BasicTS train data.

    Shape:
        train_data.npy: [T, N]
        return:
            mean: scalar tensor
            std: scalar tensor
    """

    train_data = np.load(os.path.join(data_file_path, "train_data.npy"), mmap_mode="r")
    mean = float(train_data.mean())
    std = float(train_data.std())
    if std == 0.0:
        std = 1.0
    return {
        "mean": torch.tensor(mean, dtype=torch.float32),
        "std": torch.tensor(std, dtype=torch.float32),
    }
