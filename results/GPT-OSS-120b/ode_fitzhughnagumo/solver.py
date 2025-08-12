import numpy as np
from typing import Any, Dict, List
import numba
from scipy.integrate import solve_ivp
@numba.njit
def _rk4(t0: float, t1: float, y0: np.ndarray,
         a: float, b: float, c: float, I: float,
          N: int = 400000) -> np.ndarray:
    dt = (t1 - t0) / N
    v = y0[0]
    w = y0[1]
    for _ in range(N):
        # k1
        dv1 = v - (v ** 3) / 3 - w + I
        dw1 = a * (b * v - c * w)

        # k2
        v2 = v + 0.5 * dt * dv1
        w2 = w + 0.5 * dt * dw1
        dv2 = v2 - (v2 ** 3) / 3 - w2 + I
        dw2 = a * (b * v2 - c * w2)

        # k3
        v3 = v + 0.5 * dt * dv2
        w3 = w + 0.5 * dt * dw2
        dv3 = v3 - (v3 ** 3) / 3 - w3 + I
        dw3 = a * (b * v3 - c * w3)

        # k4
        v4 = v + dt * dv3
        w4 = w + dt * dw3
        dv4 = v4 - (v4 ** 3) / 3 - w4 + I
        dw4 = a * (b * v4 - c * w4)

        # update
        v += dt * (dv1 + 2 * dv2 + 2 * dv3 + dv4) / 6.0
        w += dt * (dw1 + 2 * dw2 + 2 * dw3 + dw4) / 6.0

    return np.array([v, w])

class Solver:
    def solve(self, problem: Dict[str, Any], **kwargs) -> List[float]:
        """
        Solve the FitzHugh‑Nagumo equations using a high‑accuracy integrator.
        """
        t0 = float(problem["t0"])
        t1 = float(problem["t1"])
        y0 = np.array(problem["y0"], dtype=np.float64)
        params = problem["params"]
        a = float(params["a"])
        b = float(params["b"])
        c = float(params["c"])
        I = float(params["I"])

        # Use SciPy's adaptive RK45 integrator for high accuracy
        sol = solve_ivp(
            lambda t, y: np.array([
                y[0] - (y[0] ** 3) / 3 - y[1] + I,
                a * (b * y[0] - c * y[1])
            ]),
            [t0, t1],
            y0,
            method="RK45",
            rtol=1e-8,
            atol=1e-8,
        )
        if not sol.success:
            raise RuntimeError(f"Integration failed: {sol.message}")
        return [float(sol.y[0, -1]), float(sol.y[1, -1])]