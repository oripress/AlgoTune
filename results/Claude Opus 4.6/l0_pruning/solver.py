import numpy as np
from heapq import nlargest

class Solver:
    def solve(self, problem, **kwargs):
        v = problem["v"]
        k = problem["k"]
        n = len(v)
        
        if k >= n:
            return {"solution": list(v)}
        if k <= 0:
            return {"solution": [0.0] * n}
        
        if n <= 200:
            # Pure Python for very small inputs
            # nlargest is efficient: O(n + k log n)
            top_k = nlargest(k, range(n), key=lambda i: abs(v[i]))
            result = [0.0] * n
            for idx in top_k:
                result[idx] = v[idx]
            return {"solution": result}
        
        v_arr = np.asarray(v, dtype=np.float64)
        abs_v = np.abs(v_arr)
        idx = np.argpartition(abs_v, n - k)
        v_arr[idx[:n - k]] = 0.0
        return {"solution": v_arr.tolist()}