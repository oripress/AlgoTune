import numpy as np
from typing import Any

try:
    from simplex_proj import simplex_project_list, simplex_project_np
    HAS_CYTHON = True
except ImportError:
    HAS_CYTHON = False

class Solver:
    def solve(self, problem, **kwargs) -> Any:
        y_list = problem["y"]
        n = len(y_list)
        
        if HAS_CYTHON:
            if n <= 1000:
                return {"solution": simplex_project_list(y_list)}
            else:
                y = np.array(y_list, dtype=np.float64)
                return {"solution": simplex_project_np(y)}
        else:
            if n <= 500:
                sorted_y = sorted(y_list, reverse=True)
                cumsum = 0.0
                rho = 0
                for i in range(n):
                    cumsum += sorted_y[i]
                    if sorted_y[i] > (cumsum - 1.0) / (i + 1):
                        rho = i
                theta_sum = 0.0
                for i in range(rho + 1):
                    theta_sum += sorted_y[i]
                theta = (theta_sum - 1.0) / (rho + 1)
                result = [0.0] * n
                for i in range(n):
                    val = y_list[i] - theta
                    result[i] = val if val > 0.0 else 0.0
                return {"solution": result}
            else:
                y = np.array(y_list, dtype=np.float64)
                sorted_y = np.sort(y)[::-1]
                cumsum_y = np.cumsum(sorted_y) - 1.0
                rho = np.where(sorted_y > cumsum_y / np.arange(1, n + 1))[0][-1]
                theta = cumsum_y[rho] / (rho + 1)
                return {"solution": np.maximum(y - theta, 0.0)}