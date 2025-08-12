import numpy as np

class Solver:
    """
    Vectorized Newton‑Raphson solver for the function
        f(x, a0, a1, a2, a3, a4, a5) =
            a1 - a2*(exp((a0 + x*a3)/a5) - 1) - (a0 + x*a3)/a4 - x

    The parameters a2…a5 are expected to be provided as attributes of the
    Solver instance (e.g. self.a2, self.a3, self.a4, self.a5).  The method
    `solve` receives a problem dictionary with lists ``x0``, ``a0`` and
    ``a1`` and returns a dictionary ``{"roots": [...]}`` containing the
    Newton‑Raphson roots (or NaN where convergence fails).
    """

    # default constants – they can be overwritten after instantiation
    a2: float = 1.0
    a3: float = 1.0
    a4: float = 1.0
    a5: float = 1.0

    @staticmethod
    def _func(x, a0, a1, a2, a3, a4, a5):
        """Vectorized evaluation of f."""
        # compute (a0 + x*a3) once for efficiency
        t = a0 + x * a3
        return a1 - a2 * (np.exp(t / a5) - 1.0) - t / a4 - x

    @staticmethod
    def _fprime(x, a0, a1, a2, a3, a4, a5):
        """Vectorized evaluation of f' = df/dx."""
        # derivative of t = a0 + x*a3 is a3
        t = a0 + x * a3
        # d/dx of exp(t/a5) = (a3/a5) * exp(t/a5)
        exp_term = np.exp(t / a5)
        return -a2 * (a3 / a5) * exp_term - a3 / a4 - 1.0

    def solve(self, problem: dict, **kwargs):
        """
        Solve for the roots of the parameterised function using a
        vectorized Newton‑Raphson iteration.

        Parameters
        ----------
        problem : dict
            Must contain the keys ``"x0"``, ``"a0"``, ``"a1"`` with list
            values of equal length.

        Returns
        -------
        dict
            ``{"roots": list_of_roots}`` where each root is a float; NaN
            indicates non‑convergence.
        """
        # ------------------------------------------------------------------
        # 1. Parse input and validate dimensions
        # ------------------------------------------------------------------
        try:
            x0 = np.asarray(problem["x0"], dtype=np.float64)
            a0 = np.asarray(problem["a0"], dtype=np.float64)
            a1 = np.asarray(problem["a1"], dtype=np.float64)
        except Exception:
            # malformed input – return empty result as reference does
            return {"roots": []}

        if not (x0.shape == a0.shape == a1.shape):
            return {"roots": []}

        n = x0.size
        if n == 0:
            return {"roots": []}

        # ------------------------------------------------------------------
        # 2. Prepare constant parameters
        # ------------------------------------------------------------------
        a2 = float(getattr(self, "a2", 1.0))
        a3 = float(getattr(self, "a3", 1.0))
        a4 = float(getattr(self, "a4", 1.0))
        a5 = float(getattr(self, "a5", 1.0))

        # ------------------------------------------------------------------
        # 3. Newton‑Raphson iteration (vectorized)
        # ------------------------------------------------------------------
        max_iter = 30
        tol = 1e-10

        x = x0.copy()
        # mask of still‑active (not yet converged) indices
        active = np.full(n, True, dtype=bool)

        for _ in range(max_iter):
            if not np.any(active):
                break

            # evaluate function and derivative only where active
            fx = self._func(x[active], a0[active], a1[active],
                            a2, a3, a4, a5)
            fpx = self._fprime(x[active], a0[active], a1[active],
                               a2, a3, a4, a5)

            # protect against zero derivative – mark as non‑convergent
            zero_der = fpx == 0
            if np.any(zero_der):
                # set those entries to NaN and deactivate them
                idx = np.where(active)[0][zero_der]
                x[idx] = np.nan
                active[idx] = False
                # continue with remaining active entries
                continue

            delta = fx / fpx
            x_new = x[active] - delta

            # check convergence for each active entry
            converged = np.abs(delta) < tol
            if np.any(converged):
                # update converged entries in the full array
                idx_all = np.where(active)[0][converged]
                x[idx_all] = x_new[converged]
                # deactivate them
                active[idx_all] = False

            # update still‑active entries
            idx_still = np.where(active)[0]
            x[idx_still] = x_new[~converged]

        # ------------------------------------------------------------------
        # 4. Post‑process: any remaining active entries are considered
        #    non‑convergent → NaN
        # ------------------------------------------------------------------
        if np.any(active):
            idx = np.where(active)[0]
            x[idx] = np.nan

        # Ensure output is a plain Python list of floats (or NaN)
        roots_list = x.tolist()
        return {"roots": roots_list}