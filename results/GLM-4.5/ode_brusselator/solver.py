from typing import Any
import numpy as np
from scipy.integrate import solve_ivp
from numba import jit
import torch

# Pre-compile the function with Numba
@jit(nopython=True, fastmath=True)
def brusselator_numba(y, A, B, B_plus_1):
    X, Y = y[0], y[1]
    X2 = X * X
    X2Y = X2 * Y
    dX_dt = A + X2Y - B_plus_1 * X
    dY_dt = B * X - X2Y
    return np.array([dX_dt, dY_dt])

# Create optimized function using PyTorch JIT
@torch.jit.script
def brusselator_torch(y_tensor: torch.Tensor, A: float, B: float, B_plus_1: float) -> torch.Tensor:
    X = y_tensor[0]
    Y = y_tensor[1]
    X2 = X * X
    X2Y = X2 * Y
    dX_dt = A + X2Y - B_plus_1 * X
    dY_dt = B * X - X2Y
    return torch.stack([dX_dt, dY_dt])

class Solver:
    def solve(self, problem: dict[str, Any], **kwargs) -> Any:
        """Solve the Brusselator reaction model using optimized approaches."""
        y0 = np.array(problem["y0"], dtype=np.float64)
        t0, t1 = problem["t0"], problem["t1"]
        params = problem["params"]
        
        A = params["A"]
        B = params["B"]
        B_plus_1 = B + 1  # Pre-compute constant
        
        # Use the Numba-compiled function (still the fastest)
        def brusselator(t, y):
            return brusselator_numba(y, A, B, B_plus_1)
        
        # Set solver parameters
        rtol = 1e-8
        atol = 1e-8
        
        # Use DOP853 with optimized parameters
        sol = solve_ivp(
            brusselator,
            [t0, t1],
            y0,
            method="DOP853",
            rtol=rtol,
            atol=atol,
            dense_output=False,
            first_step=None,
            max_step=(t1 - t0) / 25,
            vectorized=False,
        )
        
        if not sol.success:
            raise RuntimeError(f"Solver failed: {sol.message}")
        
        # Extract final state
        return sol.y[:, -1]