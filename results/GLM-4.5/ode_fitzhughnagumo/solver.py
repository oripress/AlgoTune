from typing import Any
import numpy as np
from scipy.integrate import solve_ivp

# Try to import Cython, fall back to regular Python if not available
try:
    import pyximport
    pyximport.install(setup_args={"include_dirs": np.get_include()})
    from _fitzhugh_cython import fitzhugh_nagumo_cython
    USE_CYTHON = True
except ImportError:
    USE_CYTHON = False

class Solver:
    def solve(self, problem: dict[str, Any], **kwargs) -> Any:
        """Solve the FitzHugh-Nagumo neuron model."""
        y0 = np.array(problem["y0"])
        t0, t1 = problem["t0"], problem["t1"]
        params = problem["params"]
        
        # Extract parameters for faster access
        a = params["a"]
        b = params["b"]
        c = params["c"]
        I = params["I"]
        
        if USE_CYTHON:
            # Use Cython-compiled function
            def fitzhugh_nagumo(t, y):
                return fitzhugh_nagumo_cython(t, y, a, b, c, I)
        else:
            # FitzHugh-Nagumo equations as a lambda function
            fitzhugh_nagumo = lambda t, y: np.array([
                y[0] - (y[0]**3) / 3 - y[1] + I,  # dv_dt
                a * (b * y[0] - c * y[1])          # dw_dt
            ])
        
        # Set solver parameters
        rtol = 1e-8
        atol = 1e-8
        
        sol = solve_ivp(
            fitzhugh_nagumo,
            [t0, t1],
            y0,
            method="RK45",
            rtol=rtol,
            atol=atol,
            dense_output=False,
        )
        
        if not sol.success:
            raise RuntimeError(f"Solver failed: {sol.message}")
        
        # Extract final state
        return sol.y[:, -1].tolist()