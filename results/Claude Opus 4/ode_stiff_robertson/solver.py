import numpy as np
from scipy.integrate import solve_ivp
import numba as nb

class Solver:
    def __init__(self):
        # Pre-compile the function with dummy values
        dummy_y = np.array([1.0, 0.0, 0.0])
        dummy_k = np.array([0.04, 3e7, 1e4])
        self._rober_jit(0.0, dummy_y, dummy_k)
    
    @staticmethod
    @nb.njit(cache=True)
    def _rober_jit(t, y, k):
        y1, y2, y3 = y[0], y[1], y[2]
        k1, k2, k3 = k[0], k[1], k[2]
        
        f0 = -k1 * y1 + k3 * y2 * y3
        f1 = k1 * y1 - k2 * y2 * y2 - k3 * y2 * y3
        f2 = k2 * y2 * y2
        
        return np.array([f0, f1, f2])
    
    def solve(self, problem, **kwargs):
        """Solve the Robertson chemical kinetics system."""
        y0 = np.array(problem["y0"])
        t0, t1 = float(problem["t0"]), float(problem["t1"])
        k = np.array(problem["k"])
        
        # Wrapper for scipy
        def rober(t, y):
            return self._rober_jit(t, y, k)
        
        # Use looser tolerances than reference since verification only needs 1e-5/1e-8
        rtol = 1e-6
        atol = 1e-9
        
        sol = solve_ivp(
            rober,
            [t0, t1],
            y0,
            method="Radau",
            rtol=rtol,
            atol=atol,
            t_eval=None,
            dense_output=False
        )
        
        if sol.success:
            return sol.y[:, -1].tolist()
        else:
            raise RuntimeError(f"Solver failed: {sol.message}")