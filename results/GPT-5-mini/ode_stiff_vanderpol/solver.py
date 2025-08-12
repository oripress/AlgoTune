from typing import Any, Dict, List
import numpy as np
from scipy.integrate import solve_ivp

class Solver:
    def solve(self, problem: Dict[str, Any], **kwargs) -> List[float]:
        """
        Solve the Van der Pol oscillator:
            dx/dt = v
            dv/dt = mu * ((1 - x^2) * v - x)

        Inputs:
            problem: dict with keys "mu", "y0", "t0", "t1"
        Returns:
            list[float]: [x(t1), v(t1)]
        """
        # Parse inputs
        try:
            mu = float(problem["mu"])
            y0 = np.asarray(problem["y0"], dtype=float).ravel()
            t0 = float(problem["t0"])
            t1 = float(problem["t1"])
        except Exception as e:
            raise ValueError(f"Invalid problem input: {e}")

        if y0.size != 2:
            raise ValueError("y0 must be of length 2: [x0, v0]")

        # Quick return
        if t1 == t0:
            return [float(y0[0]), float(y0[1])]

        # Right-hand side
        def fun(t, y):
            x, v = y
            return [v, mu * ((1.0 - x * x) * v - x)]

        # Analytic Jacobian for stiff solvers
        def jac(t, y):
            x, v = y
            return np.array(
                [
                    [0.0, 1.0],
                    [mu * (-1.0 - 2.0 * x * v), mu * (1.0 - x * x)],
                ],
                dtype=float,
            )

        # Heuristic method selection: use stiff solver for larger mu
        method = str(kwargs.get("method", "Radau" if abs(mu) > 50.0 else "RK45"))
        rtol = float(kwargs.get("rtol", 1e-8))
        atol = float(kwargs.get("atol", 1e-9))
        jac_arg = jac if method in ("Radau", "BDF") else None

        # Primary attempt
        try:
            sol = solve_ivp(
                fun,
                (t0, t1),
                y0,
                method=method,
                jac=jac_arg,
                rtol=rtol,
                atol=atol,
                dense_output=False,
            )
        except Exception:
            sol = None

        # Fallback to Radau with analytic Jacobian
        if sol is None or not getattr(sol, "success", False):
            try:
                sol = solve_ivp(
                    fun,
                    (t0, t1),
                    y0,
                    method="Radau",
                    jac=jac,
                    rtol=1e-8,
                    atol=1e-9,
                    dense_output=False,
                )
            except Exception:
                sol = None

        if sol is None:
            raise RuntimeError("ODE solver failed to produce a solution.")

        if not sol.success:
            raise RuntimeError(f"ODE solver failed: {getattr(sol, 'message', '')}")

        y_final = sol.y[:, -1]
        return [float(y_final[0]), float(y_final[1])]