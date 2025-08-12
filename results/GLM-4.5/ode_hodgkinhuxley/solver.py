from typing import Any
import numpy as np
from scipy.integrate import solve_ivp

# Try to import Cython functions, fall back to Python if compilation fails
try:
    from hodgkin_huxley_cython import hodgkin_huxley_rhs
    USE_CYTHON = True
except ImportError:
    USE_CYTHON = False

class Solver:
    def solve(self, problem: dict[str, Any], **kwargs) -> Any:
        """Solve the Hodgkin-Huxley neuron model."""
        y0 = np.array(problem["y0"])
        t0, t1 = problem["t0"], problem["t1"]
        params = problem["params"]

        # Pre-compute parameters to avoid repeated dictionary lookups
        C_m = params["C_m"]
        g_Na = params["g_Na"]
        g_K = params["g_K"]
        g_L = params["g_L"]
        E_Na = params["E_Na"]
        E_K = params["E_K"]
        E_L = params["E_L"]
        I_app = params["I_app"]

        if USE_CYTHON:
            # Use Cython-compiled RHS function
            def hodgkin_huxley(t, y):
                return hodgkin_huxley_rhs(t, y, C_m, g_Na, g_K, g_L, E_Na, E_K, E_L, I_app)
        else:
            # Use Python RHS function
            def hodgkin_huxley(t, y):
                # Unpack state variables
                V, m, h, n = y  # V = membrane potential, m,h,n = gating variables

                # Calculate alpha and beta rate constants
                # Handle singularities in rate functions (when denominator approaches 0)
                if V == -40.0:
                    alpha_m = 1.0  # L'Hôpital's rule limit at V = -40.0
                else:
                    alpha_m = 0.1 * (V + 40.0) / (1.0 - np.exp(-(V + 40.0) / 10.0))

                V_plus_65 = V + 65.0
                beta_m = 4.0 * np.exp(-V_plus_65 / 18.0)

                alpha_h = 0.07 * np.exp(-V_plus_65 / 20.0)
                beta_h = 1.0 / (1.0 + np.exp(-(V + 35.0) / 10.0))

                # Handle singularity in alpha_n at V = -55.0
                if V == -55.0:
                    alpha_n = 0.1  # L'Hôpital's rule limit at V = -55.0
                else:
                    alpha_n = 0.01 * (V + 55.0) / (1.0 - np.exp(-(V + 55.0) / 10.0))

                beta_n = 0.125 * np.exp(-V_plus_65 / 80.0)

                # Ensure gating variables stay in [0, 1]
                m = np.clip(m, 0.0, 1.0)
                h = np.clip(h, 0.0, 1.0)
                n = np.clip(n, 0.0, 1.0)

                # Calculate ionic currents with optimized operations
                V_minus_E_Na = V - E_Na
                V_minus_E_K = V - E_K
                V_minus_E_L = V - E_L
                
                # Use power operator for better performance
                m3 = m ** 3
                n4 = n ** 4
                
                I_Na = g_Na * m3 * h * V_minus_E_Na
                I_K = g_K * n4 * V_minus_E_K
                I_L = g_L * V_minus_E_L

                # Differential equations
                dVdt = (I_app - I_Na - I_K - I_L) / C_m
                dmdt = alpha_m * (1.0 - m) - beta_m * m
                dhdt = alpha_h * (1.0 - h) - beta_h * h
                dndt = alpha_n * (1.0 - n) - beta_n * n

                return np.array([dVdt, dmdt, dhdt, dndt])

        # Set solver parameters - use DOP853 high-order method
        rtol = 1e-8
        atol = 1e-8

        method = "DOP853"
        
        sol = solve_ivp(
            hodgkin_huxley,
            [t0, t1],
            y0,
            method=method,
            rtol=rtol,
            atol=atol,
            dense_output=False,
        )

        if not sol.success:
            raise RuntimeError(f"Solver failed: {sol.message}")

        # Extract final state
        return sol.y[:, -1].tolist()