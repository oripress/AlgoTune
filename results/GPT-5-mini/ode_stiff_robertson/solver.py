from typing import Any, Dict, List
import numpy as np

class Solver:
    def solve(self, problem: Dict[str, Any], **kwargs) -> List[float]:
        """
        Solve the Robertson chemical kinetics ODE and return [y1, y2, y3] at t1.
        Uses SciPy's stiff solver (Radau) with an analytic Jacobian. Tolerances
        tuned for speed while keeping the solution within the verifier bounds.
        """
        # Local import to keep module import cheap if SciPy isn't needed elsewhere
        try:
            from scipy.integrate import solve_ivp
        except Exception as e:
            raise RuntimeError("scipy is required for this solver") from e

        y0 = np.asarray(problem["y0"], dtype=np.float64).ravel()
        if y0.size != 3:
            y0 = y0.reshape(3)
        t0 = float(problem["t0"])
        t1 = float(problem["t1"])
        k1, k2, k3 = map(float, problem["k"])

        # Quick return
        if t1 == t0:
            return [float(y0[0]), float(y0[1]), float(y0[2])]

        # Tolerances: chosen to be efficient but within verification tolerance
        rtol = float(kwargs.get("rtol", 1e-6))
        atol = float(kwargs.get("atol", 1e-9))
        method = kwargs.get("method", "Radau")

        def rober(t, y):
            y1, y2, y3 = y
            return np.array(
                [
                    -k1 * y1 + k3 * y2 * y3,
                    k1 * y1 - k2 * y2 * y2 - k3 * y2 * y3,
                    k2 * y2 * y2,
                ],
                dtype=np.float64,
            )

        def jac(t, y):
            y1, y2, y3 = y
            return np.array(
                [
                    [-k1, k3 * y3, k3 * y2],
                    [k1, -2.0 * k2 * y2 - k3 * y3, -k3 * y2],
                    [0.0, 2.0 * k2 * y2, 0.0],
                ],
                dtype=np.float64,
            )

        sol = solve_ivp(rober, (t0, t1), y0, method=method, jac=jac, rtol=rtol, atol=atol)
        if not sol.success:
            sol = solve_ivp(rober, (t0, t1), y0, method="BDF", jac=jac, rtol=rtol, atol=atol)
        if not sol.success:
            sol = solve_ivp(
                rober, (t0, t1), y0, method="LSODA", rtol=max(1e-8, rtol), atol=max(1e-10, atol)
            )
        if not sol.success:
            raise RuntimeError(f"ODE solver failed: {sol.message}")

        y_final = sol.y[:, -1]
        y1_f, y2_f, y3_f = float(y_final[0]), float(y_final[1]), float(y_final[2])

        # Clamp tiny negatives due to numerical error
        if -1e-14 < y1_f < 0.0:
            y1_f = 0.0
        if -1e-14 < y2_f < 0.0:
            y2_f = 0.0
        if -1e-14 < y3_f < 0.0:
            y3_f = 0.0

        return [y1_f, y2_f, y3_f]