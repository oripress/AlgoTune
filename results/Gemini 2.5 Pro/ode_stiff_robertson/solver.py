from typing import Any
import numpy as np
from scipy.integrate import solve_ivp

class Solver:
    def solve(self, problem: dict, **kwargs) -> Any:
        """
        Solves the Robertson chemical kinetics problem using SciPy's solve_ivp,
        which is highly robust for stiff ODEs.
        """
        t0 = problem["t0"]
        t1 = problem["t1"]
        # SciPy uses NumPy arrays. Use float64 for precision.
        y0 = np.array(problem["y0"], dtype=np.float64)
        k = problem["k"]
        k1, k2, k3 = k[0], k[1], k[2]

        # The vector field of the Robertson problem, compatible with solve_ivp.
        # Note the signature: t, y, followed by args.
        def rober(t, y, k1_arg, k2_arg, k3_arg):
            dy1 = -k1_arg * y[0] + k3_arg * y[1] * y[2]
            dy2 = k1_arg * y[0] - k2_arg * y[1]**2 - k3_arg * y[1] * y[2]
            dy3 = k2_arg * y[1]**2
            return [dy1, dy2, dy3]

        # The analytical Jacobian of the vector field, compatible with solve_ivp.
        def rober_jac(t, y, k1_arg, k2_arg, k3_arg):
            return [
                [-k1_arg, k3_arg * y[2], k3_arg * y[1]],
                [k1_arg, -2 * k2_arg * y[1] - k3_arg * y[2], -k3_arg * y[1]],
                [0.0, 2 * k2_arg * y[1], 0.0]
            ]

        # Solve the ODE system using the 'BDF' method, which is excellent for
        # stiff problems. Providing the Jacobian is critical for performance.
        sol = solve_ivp(
            fun=rober,
            t_span=[t0, t1],
            y0=y0,
            method='BDF',
            jac=rober_jac,
            args=(k1, k2, k3),
            rtol=1e-7,
            atol=1e-9,
            t_eval=[t1]  # Only request the solution at the final time point.
        )

        # sol.y is a 2D array of shape (n_states, n_timesteps).
        # Since we evaluated at only one time point, its shape is (3, 1).
        # We extract the first (and only) column and convert it to a list.
        return sol.y[:, 0].tolist()