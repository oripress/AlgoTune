import numpy as np

class Solver:
    def solve(self, problem, **kwargs):
        y = np.asarray(problem["y"], dtype=np.float64).ravel()
        n = len(y)
        
        # Sort in descending order (use negative sort for speed)
        sorted_y = -np.sort(-y)
        
        # Cumulative sum minus 1
        cumsum_y = np.cumsum(sorted_y)
        cumsum_y -= 1.0
        
        # Find rho using vectorized comparison
        # sorted_y[i] * (i+1) > cumsum_y[i]
        indices = np.arange(1, n + 1, dtype=np.float64)
        cond = sorted_y * indices - cumsum_y
        rho = np.flatnonzero(cond > 0.0)[-1]
        
        # Compute theta
        theta = cumsum_y[rho] / (rho + 1)
        
        # Project in-place
        result = y - theta
        np.maximum(result, 0.0, out=result)
        
        return {"solution": result}