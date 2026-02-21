from typing import Any
from fast_solver import rk45_solve_cython

class Solver:
    def solve(self, problem: dict[str, Any], **kwargs) -> Any:
        y0 = problem["y0"]
        t0 = float(problem["t0"])
        t1 = float(problem["t1"])
        params = problem["params"]
        
        alpha = float(params["alpha"])
        beta = float(params["beta"])
        delta = float(params["delta"])
        gamma = float(params["gamma"])
        
        rtol = 1e-9
        atol = 1e-9
        
        return rk45_solve_cython(t0, t1, float(y0[0]), float(y0[1]), alpha, beta, delta, gamma, rtol, atol)