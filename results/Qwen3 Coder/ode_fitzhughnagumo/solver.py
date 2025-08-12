import numpy as np
from scipy.integrate import solve_ivp
from typing import Any, Dict
class Solver:
    def solve(self, problem: Dict[str, Any], **kwargs) -> Any:
        """Solve the FitzHugh-Nagumo model efficiently."""
        # Extract parameters
        y0 = np.array(problem["y0"])
        t0, t1 = problem["t0"], problem["t1"]
        params = problem["params"]
        a, b, c, I = params["a"], params["b"], params["c"], params["I"]
        
        # Define the FitzHugh-Nagumo equations
        # Define the FitzHugh-Nagumo equations
        # Define the FitzHugh-Nagumo equations
        def fitzhugh_nagumo(t, y):
            v, w = y[0], y[1]
            dv_dt = v - (v**3) / 3.0 - w + I
            dw_dt = a * (b * v - c * w)
            return [dv_dt, dw_dt]
        
        # Solve the ODE
        sol = solve_ivp(
            fun=fitzhugh_nagumo,
            t_span=[t0, t1],
            y0=y0,
            method="RK45",
            rtol=1e-8,
            atol=1e-8,
            dense_output=False
        )
        
        if sol.success:
            return sol.y[:, -1].tolist()
        else:
            raise RuntimeError(f"Solver failed: {sol.message}")