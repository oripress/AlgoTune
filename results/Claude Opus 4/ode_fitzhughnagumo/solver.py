import numpy as np
from scipy.integrate import solve_ivp
from typing import Any

class Solver:
    def solve(self, problem: dict[str, Any], **kwargs) -> Any:
        y0 = np.array(problem["y0"])
        t0, t1 = problem["t0"], problem["t1"]
        params = problem["params"]
        
        def fitzhugh_nagumo(t, y):
            v, w = y
            a = params["a"]
            b = params["b"]
            c = params["c"]
            I = params["I"]
            
            dv_dt = v - (v**3) / 3 - w + I
            dw_dt = a * (b * v - c * w)
            
            return np.array([dv_dt, dw_dt])
        
        sol = solve_ivp(
            fitzhugh_nagumo,
            [t0, t1],
            y0,
            method="RK45",
            rtol=1e-8,
            atol=1e-8
        )
        
        if sol.success:
            return sol.y[:, -1].tolist()
        else:
            raise RuntimeError(f"Solver failed: {sol.message}")