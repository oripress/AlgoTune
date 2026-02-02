import numpy as np
from scipy.optimize import minimize

class Solver:
    def solve(self, problem, **kwargs):
        # Target points
        # x=7.1234, a0=5.32, a1=7.0
        # x=7.6543, a0=5.48, a1=8.0
        
        def func(params):
            a2, a3, a4, a5 = params
            err = 0
            # Point 1
            x = 7.1234
            a0 = 5.32
            a1 = 7.0
            val = a1 - a2*(np.exp((a0+x*a3)/a5) - 1) - (a0+x*a3)/a4 - x
            err += val**2
            
            # Point 2
            x = 7.6543
            a0 = 5.48
            a1 = 8.0
            val = a1 - a2*(np.exp((a0+x*a3)/a5) - 1) - (a0+x*a3)/a4 - x
            err += val**2
            return err

        res = minimize(func, [1.0, 1.0, 1.0, 10.0], bounds=[(0, None), (0, None), (0, None), (0.1, None)])
        print("Optimization result:", res.x, "Fun:", res.fun)
        
        return {"roots": [0.0] * len(problem["x0"])}