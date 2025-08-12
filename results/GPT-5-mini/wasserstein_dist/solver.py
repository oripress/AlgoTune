from typing import Any
import numpy as np

# Try to use the fast Cython implementation if present.
try:
    import solver_cy as _solver_cy  # compiled extension (optional)
except Exception:
    _solver_cy = None

class Solver:
    def solve(self, problem: dict, **kwargs) -> float:
        """
        Compute the 1D Wasserstein (Earth Mover) distance between two discrete
        distributions supported on {1,2,...,n}.

        If a compiled Cython extension (solver_cy) is available, it will be used
        for maximum performance; otherwise a numpy-vectorized O(n) fallback is used.

        The implementation normalizes each weight vector (like scipy.stats.wasserstein_distance)
        if its sum is > 0. If shapes differ, the shorter vector is padded with zeros.
        """
        try:
            u = np.asarray(problem.get("u", []), dtype=np.float64)
            v = np.asarray(problem.get("v", []), dtype=np.float64)

            # Ensure 1-D arrays
            if u.ndim != 1:
                u = u.ravel()
            if v.ndim != 1:
                v = v.ravel()

            # Pad to same length
            n = max(u.size, v.size)
            if u.size != n:
                u2 = np.zeros(n, dtype=np.float64)
                u2[: u.size] = u
                u = u2
            if v.size != n:
                v2 = np.zeros(n, dtype=np.float64)
                v2[: v.size] = v
                v = v2

            # Use Cython implementation when available for speed
            if _solver_cy is not None:
                u_c = np.ascontiguousarray(u, dtype=np.float64)
                v_c = np.ascontiguousarray(v, dtype=np.float64)
                try:
                    res = float(_solver_cy.wasserstein(u_c, v_c))
                except Exception:
                    # Fallback to numpy computation if cython call fails
                    su = u_c.sum()
                    sv = v_c.sum()
                    if su > 0.0:
                        u_n = u_c / su
                    else:
                        u_n = u_c
                    if sv > 0.0:
                        v_n = v_c / sv
                    else:
                        v_n = v_c
                    diff = np.cumsum(u_n - v_n)
                    res = float(np.sum(np.abs(diff)))
            else:
                # Pure numpy path
                su = u.sum()
                sv = v.sum()
                if su > 0.0:
                    u = u / su
                if sv > 0.0:
                    v = v / sv
                diff = np.cumsum(u - v)
                res = float(np.sum(np.abs(diff)))

            if not np.isfinite(res):
                return float(n)
            return res
        except Exception:
            # Fallback similar to reference implementation
            try:
                return float(len(problem.get("u", [])))
            except Exception:
                return 0.0