from typing import Any, Dict
import numpy as np
from scipy.integrate import solve_ivp

class Solver:
    def solve(self, problem: Dict[str, Any], **kwargs) -> Any:
        """
        Solve the Hodgkin-Huxley ODE system and return [V, m, h, n] at t1.
        Uses scipy.solve_ivp (RK45) with tight tolerances to match the reference.
        """
        y0 = np.asarray(problem["y0"], dtype=float)
        t0 = float(problem["t0"])
        t1 = float(problem["t1"])
        params = problem.get("params", {})

        # Quick return if no time evolution requested
        if t1 == t0:
            return y0.tolist()

        # Extract parameters with reasonable defaults
        C_m = float(params.get("C_m", 1.0))
        g_Na = float(params.get("g_Na", 120.0))
        g_K = float(params.get("g_K", 36.0))
        g_L = float(params.get("g_L", 0.3))
        E_Na = float(params.get("E_Na", 50.0))
        E_K = float(params.get("E_K", -77.0))
        E_L = float(params.get("E_L", -54.4))
        I_app = float(params.get("I_app", 0.0))

        def hodgkin_huxley(t, y):
            V, m, h, n = y

            # alpha_m with singularity handling near V = -40 mV
            dv40 = V + 40.0
            if abs(dv40) < 1e-12:
                alpha_m = 1.0
            else:
                alpha_m = 0.1 * dv40 / (1.0 - np.exp(-dv40 / 10.0))

            beta_m = 4.0 * np.exp(-(V + 65.0) / 18.0)

            alpha_h = 0.07 * np.exp(-(V + 65.0) / 20.0)
            beta_h = 1.0 / (1.0 + np.exp(-(V + 35.0) / 10.0))

            # alpha_n with singularity handling near V = -55 mV
            dv55 = V + 55.0
            if abs(dv55) < 1e-12:
                alpha_n = 0.1
            else:
                alpha_n = 0.01 * dv55 / (1.0 - np.exp(-dv55 / 10.0))

            beta_n = 0.125 * np.exp(-(V + 65.0) / 80.0)

            # Ensure gating variables remain in [0,1]
            m = np.clip(m, 0.0, 1.0)
            h = np.clip(h, 0.0, 1.0)
            n = np.clip(n, 0.0, 1.0)

            # Ionic currents
            I_Na = g_Na * (m ** 3) * h * (V - E_Na)
            I_K = g_K * (n ** 4) * (V - E_K)
            I_L = g_L * (V - E_L)

            # ODEs
            dVdt = (I_app - I_Na - I_K - I_L) / C_m
            dmdt = alpha_m * (1.0 - m) - beta_m * m
            dhdt = alpha_h * (1.0 - h) - beta_h * h
            dndt = alpha_n * (1.0 - n) - beta_n * n

            return np.array([dVdt, dmdt, dhdt, dndt], dtype=float)

        # Tolerances to match reference solver behavior
        rtol = 1e-8
        atol = 1e-8

        sol = solve_ivp(
            hodgkin_huxley,
            (t0, t1),
            y0,
            method="RK45",
            rtol=rtol,
            atol=atol,
            t_eval=None,
            dense_output=False,
        )

        if not sol.success:
            raise RuntimeError(f"Hodgkin-Huxley solver failed: {sol.message}")

        y_final = sol.y[:, -1]
        # Clip gating variables to valid range before returning
        y_final[1:] = np.clip(y_final[1:], 0.0, 1.0)
        return y_final.tolist()