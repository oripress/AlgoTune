from typing import Any
import numpy as np
import scipy.odr as odr

def linear_model(B, x):
    """Linear model function for ODR."""
    return B[0] * x + B[1]

class Solver:
    def solve(self, problem: dict[str, Any]) -> dict[str, Any]:
        """Fit weighted ODR with scipy.odr.
        
        NOTE: Your solution must pass validation by:
        1. Returning correctly formatted output
        2. Having no NaN or infinity values
        3. Matching expected results within numerical tolerance
        """
        # Convert to numpy arrays with explicit dtype for efficiency
        x = np.array(problem["x"], dtype=np.float64)
        y = np.array(problem["y"], dtype=np.float64)
        sx = np.array(problem["sx"], dtype=np.float64)
        sy = np.array(problem["sy"], dtype=np.float64)
        
        # Create data and model
        data = odr.RealData(x, y=y, sx=sx, sy=sy)
        model = odr.Model(linear_model)
        
        # Use ODR with optimized parameters and different method
        odr_instance = odr.ODR(data, model, beta0=[0.0, 1.0], maxit=100)
        # Try using a different method that might be faster
        odr_instance.set_job(fit_type=0)  # Use explicit method
        output = odr_instance.run()
        
        return {"beta": output.beta.tolist()}