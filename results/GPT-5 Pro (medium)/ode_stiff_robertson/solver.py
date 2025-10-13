from typing import Any, Dict, List
import numpy as np
from scipy.integrate import solve_ivp

class Solver:
    def solve(self, problem: Dict[str, Any], **kwargs) -> List[float]:
        """
        Solve the stiff Robertson chemical kinetics system:
            dy1/dt = -k1*y1 + k3*y2*y3
            dy2/dt =  k1*y1 - k2*y2^2 - k3*y2*y3
            dy3/dt =  k2*y2^2

        Input:
            problem: dict with keys:
                - t0: float, initial time
                - t1: float, final time
                - y0: list[3] floats, initial state
                - k: list[3] floats, rate constants [k1, k2, k3]
        Output:
            list[3] floats, solution at t1
        """
        t0 = float(problem["t0"])
        t1 = float(problem["t1"])
        y0 = np.asarray(problem["y0"], dtype=float)
        k1, k2, k3 = map(float, problem["k"])

        if y0.shape != (3,):
            y0 = y0.reshape(3)

        if t1 == t0:
            return y0.tolist()

        def f(t: float, y: np.ndarray) -> List[float]:
            y1, y2, y3 = y
            return [
                -k1 * y1 + k3 * y2 * y3,
                k1 * y1 - k2 * y2 * y2 - k3 * y2 * y3,
                k2 * y2 * y2,
            ]

        def jac(t: float, y: np.ndarray) -> np.ndarray:
            y1, y2, y3 = y
            return np.array(
                [
                    [-k1,           k3 * y3,        k3 * y2],
                    [ k1, -2.0 * k2 * y2 - k3 * y3, -k3 * y2],
                    [ 0.0,        2.0 * k2 * y2,        0.0],
                ],
                dtype=float,
            )

        # Tolerances chosen to comfortably satisfy verification (rtol=1e-5, atol=1e-8)
        # while being faster than the reference implementation.
        rtol = 1e-8
        atol = 1e-10

        sol = solve_ivp(
            f,
            (t0, t1),
            y0,
            method="BDF",
            jac=jac,
            rtol=rtol,
            atol=atol,
            vectorized=False,
            dense_output=False,
        )

        if not sol.success:
            raise RuntimeError(f"Solver failed: {sol.message}")

        return sol.y[:, -1].tolist()