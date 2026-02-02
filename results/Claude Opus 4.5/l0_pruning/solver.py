import numpy as np
from numba import njit

@njit(cache=True, fastmath=True)
def _apply_threshold(v, threshold, k, n):
    # Count elements above and equal to threshold
    above_count = 0
    total_equal = 0
    for i in range(n):
        av = abs(v[i])
        if av > threshold:
            above_count += 1
        elif av == threshold:
            total_equal += 1
    
    need_equal = k - above_count
    skip_equal = total_equal - need_equal
    
    # Build result in a single pass
    result = np.zeros(n, dtype=np.float64)
    equal_seen = 0
    
    for i in range(n):
        av = abs(v[i])
        if av > threshold:
            result[i] = v[i]
        elif av == threshold:
            if equal_seen >= skip_equal:
                result[i] = v[i]
            equal_seen += 1
    
    return result

class Solver:
    def __init__(self):
        dummy_v = np.array([1.0, 2.0, 3.0, 0.5])
        _apply_threshold(dummy_v, 1.5, 2, 4)

    def solve(self, problem, **kwargs):
        v = np.asarray(problem["v"], dtype=np.float64).ravel()
        k = int(problem["k"])
        n = v.size
        
        if k >= n:
            return {"solution": v.tolist()}
        if k <= 0:
            return {"solution": [0.0] * n}
        
        abs_v = np.abs(v)
        threshold = np.partition(abs_v, n - k)[n - k]
        
        result = _apply_threshold(v, threshold, k, n)
        return {"solution": result.tolist()}