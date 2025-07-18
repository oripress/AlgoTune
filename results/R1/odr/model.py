import numpy as np
from numba import njit

@njit
def linear_model(B, x):
    return B[0] * x + B[1]