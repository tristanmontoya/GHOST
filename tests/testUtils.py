import numpy as np


def gaussian_test(x: np.ndarray) -> float:
    # x should be d x 1
    return np.exp(x.T @ x)
