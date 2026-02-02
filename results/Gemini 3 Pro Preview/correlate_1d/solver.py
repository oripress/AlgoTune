import numpy as np
from scipy import signal
from numba import njit

@njit(fastmath=True, cache=True)
def correlate_valid(a, b):
    n = a.shape[0]
    m = b.shape[0]
    L = n - m + 1
    res = np.empty(L, dtype=np.float64)
    for i in range(L):
        acc = 0.0
        for j in range(m):
            acc += a[i + j] * b[j]
        res[i] = acc
    return res

@njit(fastmath=True, cache=True)
def correlate_full(a, b):
    n = a.shape[0]
    m = b.shape[0]
    L = n + m - 1
    res = np.zeros(L, dtype=np.float64)
    for k in range(L):
        start_j = max(0, k - m + 1)
        end_j = min(n - 1, k)
        acc = 0.0
        for j in range(start_j, end_j + 1):
            acc += a[j] * b[m - 1 - k + j]
        res[k] = acc
    return res

class Solver:
    def __init__(self):
        self.mode = "full"
        # Warmup
        a = np.array([1., 2.], dtype=np.float64)
        b = np.array([1.], dtype=np.float64)
        correlate_valid(a, b)
        correlate_full(a, b)

    def solve(self, problem: list) -> list:
        results = []
        mode = self.mode
        is_valid = (mode == "valid")
        
        # Threshold determined heuristically
        THRESHOLD = 10000 
        
        for a, b in problem:
            n = a.shape[0]
            m = b.shape[0]
            
            if is_valid:
                if m > n:
                    continue
                if n * m < THRESHOLD:
                    res = correlate_valid(a, b)
                else:
                    res = signal.correlate(a, b, mode="valid")
            else:
                if n * m < THRESHOLD:
                    res = correlate_full(a, b)
                else:
                    res = signal.correlate(a, b, mode="full")
            results.append(res)
        return results