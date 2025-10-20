import random
import numpy as np
import torch


def set_seed(seed: int, force_deterministic: bool = False) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    if force_deterministic:
        torch.backends.cudnn.deterministic = True
