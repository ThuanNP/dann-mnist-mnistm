"""Cố định random seed để kết quả tái lập được."""
import os
import random

import numpy as np
import torch


def set_seed(seed: int) -> torch.Generator:
    """Cố định seed cho random, NumPy, PyTorch; trả về Generator cho DataLoader."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    g = torch.Generator()
    g.manual_seed(seed)
    return g


def seed_worker(worker_id: int) -> None:
    """Seed cho từng worker của DataLoader."""
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)
