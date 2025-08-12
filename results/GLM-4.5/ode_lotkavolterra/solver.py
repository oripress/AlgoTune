from typing import Any
import numpy as np
from scipy.integrate import solve_ivp

# Try to import Cython version, fall back to Python if not available
try:
    import pyximport
    pyximport.install(setup_args={"include_dirs": np.get_include()})
    from _lotka_volterra import lotka_volterra_cython
    USE_CYTHON = True
except ImportError:
    USE_CYTHON = False

class Solver:
    def solve(self, problem: dict[str, Any], **kwargs) -> Any:
        """Your implementation goes here."""
        # Minimal variable creation for speed
        y0 = np.array(problem["y0"])
        t_span = [problem["t0"], problem["t1"]]
        p = problem["params"]
        
        if USE_CYTHON:
            # Use Cython version
            def lotka_volterra(t, y):
                return lotka_volterra_cython(t, y, p["alpha"], p["beta"], p["delta"], p["gamma"])
        else:
            # Inline parameter access for maximum speed
            def lotka_volterra(t, y):
                x, y_val = y
                return np.array([
                    p["alpha"] * x - p["beta"] * x * y_val,
                    p["delta"] * x * y_val - p["gamma"] * y_val
                ])

        # Use LSODA with slightly relaxed tolerances for speed
        sol = solve_ivp(
            lotka_volterra,
            t_span,
            y0,
            method="LSODA",
            rtol=1e-9,  # Slightly relaxed from 1e-10
            atol=1e-9,  # Slightly relaxed from 1e-10
            dense_output=False,
            first_step=1e-2,  # Start with even larger step
            max_step=1000.0,  # Allow very large steps for long integrations
        )

        if not sol.success:
            raise RuntimeError(f"Solver failed: {sol.message}")
        
        return sol.y[:, -1].tolist()