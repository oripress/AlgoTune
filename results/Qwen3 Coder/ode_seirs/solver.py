class Solver:
    def solve(self, problem, **kwargs):
        """Solve the SEIRS model for the given problem."""
        import numpy as np
        from scipy.integrate import solve_ivp
        
        # Extract parameters
        y0 = problem["y0"]
        t0, t1 = problem["t0"], problem["t1"]
        params = problem["params"]
        
        beta = params["beta"]
        sigma = params["sigma"]
        gamma = params["gamma"]
        omega = params["omega"]

        # SEIRS model function
        def seirs(t, y):
            S, E, I, R = y
            dSdt = -beta * S * I + omega * R
            dEdt = beta * S * I - sigma * E
            dIdt = sigma * E - gamma * I
            dRdt = gamma * I - omega * R
            return [dSdt, dEdt, dIdt, dRdt]

        # Use a fixed-step approach with a coarse grid
        sol = solve_ivp(
            seirs,
            [t0, t1],
            y0,
            method="LSODA",
            rtol=1e-3,
            atol=1e-6,
            t_eval=[t1],
            dense_output=False,
        )

        # Extract final state
        if sol.success:
            return sol.y[:, -1].tolist()
        else:
            raise RuntimeError(f"Solver failed: {sol.message}")