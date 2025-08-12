import numpy as np

class Solver:
    def __init__(self):
        # Fixed parameters - these might need adjustment
        self.a2 = 1.0
        self.a3 = 1.0
        self.a4 = 10.0
        self.a5 = 0.5
        
    def solve(self, problem: dict) -> dict:
        """Fast vectorized Newton-Raphson solver"""
        try:
            # Convert to numpy arrays
            x = np.asarray(problem["x0"], dtype=np.float64)
            a0 = np.asarray(problem["a0"], dtype=np.float64)
            a1 = np.asarray(problem["a1"], dtype=np.float64)
            n = len(x)
            
            if len(a0) != n or len(a1) != n:
                raise ValueError("Input arrays have mismatched lengths")
            
            # Pre-compute constants
            a2, a3, a4, a5 = self.a2, self.a3, self.a4, self.a5
            a3_over_a5 = a3 / a5
            a3_over_a4 = a3 / a4
            
            # Newton-Raphson parameters
            max_iter = 50
            tol = 1e-10
            
            # Main iteration loop
            for _ in range(max_iter):
                # Compute (a0 + x*a3)/a5 once
                arg = (a0 + x * a3) / a5
                exp_arg = np.exp(arg)
                
                # Function value: a1 - a2*(exp((a0+x*a3)/a5) - 1) - (a0+x*a3)/a4 - x
                f = a1 - a2 * (exp_arg - 1) - (a0 + x * a3) / a4 - x
                
                # Derivative: -a2 * (a3/a5) * exp((a0+x*a3)/a5) - a3/a4 - 1
                fp = -a2 * a3_over_a5 * exp_arg - a3_over_a4 - 1
                
                # Newton step
                delta = f / fp
                x -= delta
                
                # Check convergence
                if np.all(np.abs(delta) < tol):
                    break
            
            # Handle any NaN or infinity values
            x[~np.isfinite(x)] = np.nan
            
            return {"roots": x.tolist()}
            
        except Exception as e:
            # Return NaN for all values on error
            n = len(problem.get("x0", []))
            return {"roots": [float("nan")] * n}