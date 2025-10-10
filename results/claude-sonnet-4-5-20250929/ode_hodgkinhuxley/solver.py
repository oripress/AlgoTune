import numpy as np
from scipy.integrate import solve_ivp
from hh_cython import hodgkin_huxley_cython

class Solver:
    def solve(self, problem, **kwargs):
        y0 = np.array(problem["y0"])
        t0, t1 = problem["t0"], problem["t1"]
        params = problem["params"]
        
        # Extract parameters
        C_m = params["C_m"]
        g_Na = params["g_Na"]
        g_K = params["g_K"]
        g_L = params["g_L"]
        E_Na = params["E_Na"]
        E_K = params["E_K"]
        E_L = params["E_L"]
        I_app = params["I_app"]
        
        # Wrapper function for solve_ivp
        def ode_func(t, y):
            return hodgkin_huxley_cython(t, y, C_m, g_Na, g_K, g_L, E_Na, E_K, E_L, I_app)
        
        # Use DOP853
        sol = solve_ivp(
            ode_func,
            [t0, t1],
            y0,
            method="DOP853",
            rtol=1e-7,
            atol=1e-8,
        )
        
        if not sol.success:
            raise RuntimeError(f"Solver failed: {sol.message}")
        
        return sol.y[:, -1].tolist()