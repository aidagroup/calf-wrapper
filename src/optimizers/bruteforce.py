import numpy as np
from typing import Callable, Optional


def optimize(
    func: Callable[[float], float],
    search_space: Optional[np.ndarray] = None,
) -> float:
    """
    Optimizes a function by evaluating it on evenly spaced points.

    Args:
        func: Function to optimize, takes a float and returns a float

    Returns:
        The argument that maximizes the function
    """
    if search_space is None:
        search_space = np.linspace(0, 1, 21)
    values = [func(x) for x in search_space]
    best_idx = np.argmax(values)
    return search_space[best_idx]
