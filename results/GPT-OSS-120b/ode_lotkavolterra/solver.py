import numpy as np
from scipy.integrate import solve_ivp
class Solver:
    @staticmethod
    def _lotka_volterra(y, params):
        """RHS of the Lotka‑Volterra system."""
        x, pred = y[0], y[1]
        alpha, beta, delta, gamma = params
        dx = alpha * x - beta * x * pred
        dy = delta * x * pred - gamma * pred
        return np.array([dx, dy], dtype=np.float64)

    @staticmethod
    def _rk4_step(y, dt, params):
        """One RK4 step."""
        k1 = Solver._lotka_volterra(y, params)
        k2 = Solver._lotka_volterra(y + 0.5 * dt * k1, params)
        k3 = Solver._lotka_volterra(y + 0.5 * dt * k2, params)
        k4 = Solver._lotka_volterra(y + dt * k3, params)
        return y + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)

    @staticmethod
    def _integrate(y0, params, t0, t1, steps=None):
        """Integrate from t0 to t1 using a fast fixed‑step RK4 scheme.
        The `steps` argument is kept for compatibility but ignored."""
        # Unpack parameters for speed
        alpha, beta, delta, gamma = params

        # Determine number of steps: at least 2000, roughly 50 per time unit
        interval = t1 - t0
        steps = max(2000, int(interval * 50))
        dt = interval / steps
        x, pred = float(y0[0]), float(y0[1])

        for _ in range(steps):
            # k1
            k1x = alpha * x - beta * x * pred
            k1y = delta * x * pred - gamma * pred

            # k2
            x2 = x + 0.5 * dt * k1x
            p2 = pred + 0.5 * dt * k1y
            k2x = alpha * x2 - beta * x2 * p2
            k2y = delta * x2 * p2 - gamma * p2

            # k3
            x3 = x + 0.5 * dt * k2x
            p3 = pred + 0.5 * dt * k2y
            k3x = alpha * x3 - beta * x3 * p3
            k3y = delta * x3 * p3 - gamma * p3

            # k4
            x4 = x + dt * k3x
            p4 = pred + dt * k3y
            k4x = alpha * x4 - beta * x4 * p4
            k4y = delta * x4 * p4 - gamma * p4

            # Update state
            x += (dt / 6.0) * (k1x + 2 * k2x + 2 * k3x + k4x)
            pred += (dt / 6.0) * (k1y + 2 * k2y + 2 * k3y + k4y)

        # Ensure non‑negative populations and return as NumPy array
        return np.maximum(np.array([x, pred], dtype=np.float64), 0.0)
        y = y0.astype(np.float64, copy=True)

        for _ in range(steps):
            x, pred = y

            # k1
            dx1 = alpha * x - beta * x * pred
            dy1 = delta * x * pred - gamma * pred
            k1 = np.array([dx1, dy1], dtype=np.float64)

            # k2
            x2 = x + 0.5 * dt * dx1
            p2 = pred + 0.5 * dt * dy1
            dx2 = alpha * x2 - beta * x2 * p2
            dy2 = delta * x2 * p2 - gamma * p2
            k2 = np.array([dx2, dy2], dtype=np.float64)

            # k3
            x3 = x + 0.5 * dt * dx2
            p3 = pred + 0.5 * dt * dy2
            dx3 = alpha * x3 - beta * x3 * p3
            dy3 = delta * x3 * p3 - gamma * p3
            k3 = np.array([dx3, dy3], dtype=np.float64)

            # k4
            x4 = x + dt * dx3
            p4 = pred + dt * dy3
            dx4 = alpha * x4 - beta * x4 * p4
            dy4 = delta * x4 * p4 - gamma * p4
            k4 = np.array([dx4, dy4], dtype=np.float64)

            # Update state
            y = y + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)

        # Ensure non‑negative populations
        return np.maximum(y, 0.0)

    def solve(self, problem, **kwargs):
        """
        Fast integration of the Lotka‑Volterra system.
        Returns the state [x, y] at the final time.
        """
        # Extract problem data
        t0 = float(problem["t0"])
        t1 = float(problem["t1"])
        y0 = np.array(problem["y0"], dtype=np.float64)

        p = problem["params"]
        params = np.array([p["alpha"], p["beta"], p["delta"], p["gamma"]], dtype=np.float64)

        # Perform integration (SciPy adaptive RK45)
        final_state = Solver._integrate(y0, params, t0, t1)

        # Ensure output is a plain Python list of floats
        return final_state.tolist()