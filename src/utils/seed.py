import random

import numpy as np

RANDOM_STATE = 42


def set_seed(seed: int = RANDOM_STATE) -> None:
    """Seed Python, NumPy, and pandas for reproducible runs."""
    random.seed(seed)
    np.random.seed(seed)
