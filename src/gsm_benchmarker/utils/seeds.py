"""Random seed utilities.

This module provides a small helper to set seeds across the main RNGs used
in experiments (Python's random, NumPy and PyTorch). It also offers an
option to force deterministic cuDNN behaviour when required for
reproducibility.
"""

import random
import numpy as np
import torch


def set_seed(seed: int, force_deterministic: bool = False) -> None:
    """Set random seeds for reproducibility.

    Parameters
    ----------
    seed:
        Integer seed to use for all RNGs.
    force_deterministic:
        If True, set PyTorch's cuDNN backend to deterministic mode. Note
        that this may reduce performance.
    """

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    if force_deterministic:
        torch.backends.cudnn.deterministic = True
