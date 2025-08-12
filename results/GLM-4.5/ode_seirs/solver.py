from typing import Any
import numpy as np
from scipy.integrate import solve_ivp
from numba import jit

try:
    from _seirs_cython import seirs_cython
    USE_CYTHON = True
except ImportError:
    USE_CYTHON = False

@jit(nopython=True, fastmath=True)
def seirs_numba(t, y, beta, sigma, gamma, omega):
    S, E, I, R = y
    beta_SI = beta * S * I
    dSdt = -beta_SI + omega * R
    dEdt = beta_SI - sigma * E
    dIdt = sigma * E - gamma * I
    dRdt = gamma * I - omega * R
    return np.array([dSdt, dEdt, dIdt, dRdt])

class Solver:
    def solve(self, problem: dict[str, Any], **kwargs) -> Any:
        """Solve the SEIRS epidemic model."""
        y0 = np.array(problem["y0"])
        t0, t1 = problem["t0"], problem["t1"]
        params = problem["params"]
        
        # Extract parameters for faster access
        beta = params["beta"]
        sigma = params["sigma"]
        gamma = params["gamma"]
        omega = params["omega"]
        
        if USE_CYTHON:
            # Use Cython-compiled function
            def seirs(t, y):
                return seirs_cython(t, y, beta, sigma, gamma, omega)
        else:
            # Use Numba-compiled function
            def seirs(t, y):
                return seirs_numba(t, y, beta, sigma, gamma, omega)
        
        # Use LSODA method with optimized parameters - try even more aggressive settings
        sol = solve_ivp(
            seirs,
            [t0, t1],
            y0,
            method="LSODA",
            rtol=1e-6,  # Slightly relaxed tolerance for speed
            atol=1e-6,
            dense_output=False,
            first_step=0.5,  # Larger initial step
            max_step=min(100.0, (t1 - t0) / 5),  # Even larger max step
            min_step=1e-6  # Prevent too small steps
        )
        
        if not sol.success:
            raise RuntimeError(f"Solver failed: {sol.message}")
        
        return sol.y[:, -1].tolist()