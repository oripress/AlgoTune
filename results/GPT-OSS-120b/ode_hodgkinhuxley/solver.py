import numpy as np
from scipy.integrate import solve_ivp
from typing import Any, Dict, List

class Solver:
    def solve(self, problem: Dict[str, Any]) -> List[float]:
        """
        Solve the Hodgkin‑Huxley ODE system from t0 to t1 and return the final state.
        The implementation follows the reference model but uses modest tolerances
        for speed while keeping the required accuracy.
        """
        # Extract data
        y0 = np.asarray(problem["y0"], dtype=float)
        t0, t1 = float(problem["t0"]), float(problem["t1"])
        params = problem["params"]

        # Pre‑extract parameters for fast access inside the ODE function
        C_m = float(params["C_m"])
        g_Na = float(params["g_Na"])
        g_K = float(params["g_K"])
        g_L = float(params["g_L"])
        E_Na = float(params["E_Na"])
        E_K = float(params["E_K"])
        E_L = float(params["E_L"])
        I_app = float(params["I_app"])

        # Define the Hodgkin‑Huxley dynamics
        def hodgkin_huxley(t, y):
            V, m, h, n = y

            # Rate constants (vectorised for speed)
            # α_m, β_m
            if np.isclose(V, -40.0, atol=1e-12):
                alpha_m = 1.0
            else:
                alpha_m = 0.1 * (V + 40.0) / (1.0 - np.exp(-(V + 40.0) / 10.0))
            beta_m = 4.0 * np.exp(-(V + 65.0) / 18.0)

            # α_h, β_h
            alpha_h = 0.07 * np.exp(-(V + 65.0) / 20.0)
            beta_h = 1.0 / (1.0 + np.exp(-(V + 35.0) / 10.0))

            # α_n, β_n
            if np.isclose(V, -55.0, atol=1e-12):
                alpha_n = 0.1
            else:
                alpha_n = 0.01 * (V + 55.0) / (1.0 - np.exp(-(V + 55.0) / 10.0))
            beta_n = 0.125 * np.exp(-(V + 65.0) / 80.0)

            # Clamp gating variables to [0, 1] (prevents drift)
            m_cl = np.clip(m, 0.0, 1.0)
            h_cl = np.clip(h, 0.0, 1.0)
            n_cl = np.clip(n, 0.0, 1.0)

            # Ionic currents
            I_Na = g_Na * (m_cl ** 3) * h_cl * (V - E_Na)
            I_K = g_K * (n_cl ** 4) * (V - E_K)
            I_L = g_L * (V - E_L)

            # Differential equations
            dVdt = (I_app - I_Na - I_K - I_L) / C_m
            dmdt = alpha_m * (1.0 - m_cl) - beta_m * m_cl
            dhdt = alpha_h * (1.0 - h_cl) - beta_h * h_cl
            dndt = alpha_n * (1.0 - n_cl) - beta_n * n_cl

            return [dVdt, dmdt, dhdt, dndt]

        # Solver tolerances – set to match reference accuracy
        rtol = 1e-8
        atol = 1e-8
        # Integrate from t0 to t1
        sol = solve_ivp(
            hodgkin_huxley,
            [t0, t1],
            y0,
            method="RK45",
            rtol=rtol,
            atol=atol,
            dense_output=False,
        )

        if not sol.success:
            raise RuntimeError(f"Integration failed: {sol.message}")

        # Return the final state (last column of y)
        return sol.y[:, -1].tolist()