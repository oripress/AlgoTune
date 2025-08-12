import numpy as np
from scipy.optimize import newton

class Solver:
    a2 = 1e-7
    a3 = 0.04
    a4 = 1000.0
    a5 = 0.02585
    def func(self, x, a0, a1, a2, a3, a4, a5):
        y = (a0 + a3 * x) / a5
        return a1 - a2 * (np.exp(y) - 1.0) - (a0 + a3 * x) / a4 - x

    def fprime(self, x, a0, a1, a2, a3, a4, a5):
        y = (a0 + a3 * x) / a5
        return -a2 * (a3 / a5) * np.exp(y) - (a3 / a4) - 1.0

    def solve(self, problem: dict, **kwargs) -> dict:
        """
        Vectorized Newton-Raphson solver for f(x,a0..a5):
          f = a1 - a2*(exp((a0 + a3*x)/a5)-1) - (a0 + a3*x)/a4 - x
        :param problem: dict with lists "x0","a0","a1"
        :param kwargs: fixed params a2,a3,a4,a5
        :return: {"roots": [root_i]} using NaN on failure
        """
        # parse variable arrays
        try:
            x0 = np.array(problem["x0"], dtype=float)
            a0 = np.array(problem["a0"], dtype=float)
            a1 = np.array(problem["a1"], dtype=float)
        except Exception:
            return {"roots": []}

        # check sizes
        n = x0.size
        if a0.size != n or a1.size != n:
            return {"roots": []}

        # fixed parameters: kwargs override defaults
        try:
            a2 = float(kwargs.get("a2", self.a2))
            a3 = float(kwargs.get("a3", self.a3))
            a4 = float(kwargs.get("a4", self.a4))
            a5 = float(kwargs.get("a5", self.a5))
        except Exception:
            return {"roots": []}

        # Newton-Raphson iterations
        x = x0.astype(float)
        tol = 1e-12
        max_iter = 50
        for _ in range(max_iter):
            y = (a0 + a3 * x) / a5
            f = a1 - a2 * (np.exp(y) - 1.0) - (a0 + a3 * x) / a4 - x
            df = -a2 * (a3 / a5) * np.exp(y) - (a3 / a4) - 1.0
            dx = f / df
            x -= dx
            if np.all(np.abs(dx) < tol):
                break

        # return roots list
        return {"roots": x.tolist()}