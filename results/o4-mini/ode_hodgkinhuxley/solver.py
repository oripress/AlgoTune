from typing import Any
import numpy as np
from scipy.integrate import solve_ivp

class Solver:
    def solve(self, problem: dict[str, Any], **kwargs) -> Any:
        # Unpack inputs
        y0 = np.array(problem["y0"], dtype=float)
        t0 = float(problem["t0"])
        t1 = float(problem["t1"])
        params = problem["params"]

        # Hodgkin-Huxley system
        def hodgkin_huxley(t, y):
            V, m, h, n = y
            C_m = params["C_m"]
            g_Na = params["g_Na"]
            g_K  = params["g_K"]
            g_L  = params["g_L"]
            E_Na = params["E_Na"]
            E_K  = params["E_K"]
            E_L  = params["E_L"]
            I_app = params["I_app"]

            # Rate constants
            if V == -40.0:
                alpha_m = 1.0
            else:
                alpha_m = 0.1 * (V + 40.0) / (1.0 - np.exp(-(V + 40.0) / 10.0))
            beta_m = 4.0 * np.exp(-(V + 65.0) / 18.0)

            alpha_h = 0.07 * np.exp(-(V + 65.0) / 20.0)
            beta_h = 1.0 / (1.0 + np.exp(-(V + 35.0) / 10.0))

            if V == -55.0:
                alpha_n = 0.1
            else:
                alpha_n = 0.01 * (V + 55.0) / (1.0 - np.exp(-(V + 55.0) / 10.0))
            beta_n = 0.125 * np.exp(-(V + 65.0) / 80.0)

            # Clamp gating variables
            m_clamped = np.clip(m, 0.0, 1.0)
            h_clamped = np.clip(h, 0.0, 1.0)
            n_clamped = np.clip(n, 0.0, 1.0)

            # Currents
            I_Na = g_Na * m_clamped**3 * h_clamped * (V - E_Na)
            I_K  = g_K  * n_clamped**4       * (V - E_K)
            I_L  = g_L  * (V - E_L)

            # Derivatives
            dVdt = (I_app - I_Na - I_K - I_L) / C_m
            dmdt = alpha_m * (1.0 - m_clamped) - beta_m * m_clamped
            dhdt = alpha_h * (1.0 - h_clamped) - beta_h * h_clamped
            dndt = alpha_n * (1.0 - n_clamped) - beta_n * n_clamped

            return [dVdt, dmdt, dhdt, dndt]

        # Solve ODE
        sol = solve_ivp(
            fun=hodgkin_huxley,
            t_span=(t0, t1),
            y0=y0,
            method="RK45",
            rtol=1e-8,
            atol=1e-8,
            t_eval=None,
            dense_output=False,
        )
        if not sol.success:
            raise RuntimeError(f"Solver failed: {sol.message}")

        # Return final state
        return sol.y[:, -1].tolist()