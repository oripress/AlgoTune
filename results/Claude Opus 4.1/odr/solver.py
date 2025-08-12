import numpy as np
from typing import Any
import scipy.odr as odr

class Solver:
    def __init__(self):
        # Pre-create the model function to avoid recreation
        self.linear_func = lambda B, x: B[0] * x + B[1]
        self.model = odr.Model(self.linear_func)
    
    def solve(self, problem: dict[str, Any]) -> dict[str, Any]:
        """Fit weighted ODR using scipy.odr optimized."""
        x = np.asarray(problem["x"], dtype=np.float64)
        y = np.asarray(problem["y"], dtype=np.float64)
        sx = np.asarray(problem["sx"], dtype=np.float64)
        sy = np.asarray(problem["sy"], dtype=np.float64)
        
        # Create data object
        data = odr.RealData(x, y=y, sx=sx, sy=sy)
        
        # Run ODR with good initial guess to speed convergence
        # Use simple weighted least squares for initial guess
        w = 1.0 / (sy * sy)
        w_sum = np.sum(w)
        x_mean = np.sum(w * x) / w_sum
        y_mean = np.sum(w * y) / w_sum
        
        numerator = np.sum(w * (x - x_mean) * (y - y_mean))
        denominator = np.sum(w * (x - x_mean) ** 2)
        slope_init = numerator / denominator
        intercept_init = y_mean - slope_init * x_mean
        
        # Create ODR object with better initial guess and run
        odr_obj = odr.ODR(data, self.model, beta0=[slope_init, intercept_init])
        output = odr_obj.run()
        
        return {"beta": output.beta.tolist()}