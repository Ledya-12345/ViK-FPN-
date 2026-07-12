

import os
import random

import numpy as np
import torch

DEFAULT_SEED = 42  # Table 3: "Random seed: 42"


def seed_everything(seed: int = DEFAULT_SEED, deterministic: bool = True) -> None:
   
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = deterministic
    torch.backends.cudnn.benchmark = True


def worker_init_fn(worker_id: int, base_seed: int = DEFAULT_SEED) -> None:
   
    worker_seed = (base_seed + worker_id) % (2 ** 32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)
