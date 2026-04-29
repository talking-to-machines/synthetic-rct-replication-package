import random

import numpy as np

RANDOM_STATE = 42


def set_seed(seed: int = RANDOM_STATE) -> None:
    """Seed Python's `random` and NumPy's RNG for reproducible runs.

    Call once at process start before any randomised operation (shuffles,
    sampling) so results are deterministic across reruns.

    Args:
        seed: Integer seed shared between the two RNGs. Defaults to
            `RANDOM_STATE` (42).
    """
    random.seed(seed)
    np.random.seed(seed)
