import numpy as np
from scipy.integrate import solve_ivp

class Solver:
    def solve(self, problem, **kwargs):
        """Solve the Hodgkin-Huxley model using optimized numerical methods."""
        # Extract parameters
        t0, t1 = problem["t0"], problem["t1"]
        y0 = np.array(problem["y0"], dtype=np.float64)
        params = problem["params"]
        
        # Define the Hodgkin-Huxley equations
        def hodgkin_huxley(t, y):
            V, m, h, n = y

            # Parameters
            C_m = params["C_m"]
            g_Na = params["g_Na"]
            g_K = params["g_K"]
            g_L = params["g_L"]
            E_Na = params["E_Na"]
            E_K = params["E_K"]
            E_L = params["E_L"]
            I_app = params["I_app"]

            # Voltage-dependent rate constants with numerical stability improvements
            # Alpha and beta for m
            V_plus_40 = V + 40.0
            if abs(V_plus_40) < 1e-10:
                alpha_m = 1.0  # L'Hôpital's rule limit
            else:
                alpha_m = 0.1 * V_plus_40 / (1.0 - np.exp(-V_plus_40 / 10.0))
            beta_m = 4.0 * np.exp(-(V + 65.0) / 18.0)

            # Alpha and beta for h
            alpha_h = 0.07 * np.exp(-(V + 65.0) / 20.0)
            beta_h = 1.0 / (1.0 + np.exp(-(V + 35.0) / 10.0))

            # Alpha and beta for n
            V_plus_55 = V + 55.0
            if abs(V_plus_55) < 1e-10:
                alpha_n = 0.1  # L'Hôpital's rule limit
            else:
                alpha_n = 0.01 * V_plus_55 / (1.0 - np.exp(-V_plus_55 / 10.0))
            beta_n = 0.125 * np.exp(-(V + 65.0) / 80.0)

            # Ensure gating variables stay in [0, 1] (clipping for numerical stability)
            m = np.clip(m, 0.0, 1.0)
            h = np.clip(h, 0.0, 1.0)
            n = np.clip(n, 0.0, 1.0)
            
            # Ionic currents
            I_Na = g_Na * m**3 * h * (V - E_Na)
            I_K = g_K * n**4 * (V - E_K)
            I_L = g_L * (V - E_L)

            # Differential equations
            dVdt = (I_app - I_Na - I_K - I_L) / C_m
            dmdt = alpha_m * (1.0 - m) - beta_m * m
            dhdt = alpha_h * (1.0 - h) - beta_h * h
            dndt = alpha_n * (1.0 - n) - beta_n * n

            return np.array([dVdt, dmdt, dhdt, dndt])

        # Solve with optimized settings for speed/accuracy balance
        sol = solve_ivp(
            hodgkin_huxley,
            [t0, t1],
            y0,
            method='LSODA',  # More efficient for stiff problems like HH
            rtol=1e-8,
            atol=1e-8,
            max_step=(t1-t0)/1000
        )

        if sol.success:
            return sol.y[:, -1].tolist()
        else:
            raise RuntimeError(f"Solver failed: {sol.message}")