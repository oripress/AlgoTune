import numpy as np
from numba import njit

@njit
def dct1_2d(problem):
    n = problem.shape[0] - 1
    m = problem.shape[1] - 1
    result = np.empty_like(problem)
    
    # Precompute cosine values
    cos_i = np.zeros((n+1, n+1))
    for i in range(n+1):
        for k in range(n+1):
            if n > 0:
                cos_i[i, k] = np.cos(np.pi * i * k / n)
            else:
                cos_i[i, k] = 1.0
    
    cos_j = np.zeros((m+1, m+1))
    for j in range(m+1):
        for l in range(m+1):
            if m > 0:
                cos_j[j, l] = np.cos(np.pi * j * l / m)
            else:
                cos_j[j, l] = 1.0
    
    # Compute 2D DCT
    for i in range(n+1):
        for j in range(m+1):
            temp = 0.0
            for k in range(n+1):
                for l in range(m+1):
                    temp += problem[k, l] * cos_i[i, k] * cos_j[j, l]
            result[i, j] = temp
    
    return result