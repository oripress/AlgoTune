from typing import Any, Dict, List

import numpy as np

try:
    # SciPy is generally faster and robust for LPs with box constraints using HiGHS
    from scipy.optimize import linprog

    _HAS_SCIPY = True
except Exception:  # pragma: no cover
    _HAS_SCIPY = False

class Solver:
    def solve(self, problem, **kwargs) -> Any:
        """
        Solve the LP box problem:
            minimize    c^T x
            subject to  A x <= b
                        0 <= x <= 1

        Parameters
        ----------
        problem : dict
            Keys:
              - "c": list of length n
              - "A": list[list] with shape (m, n) or empty
              - "b": list of length m or empty

        Returns
        -------
        dict
            {"solution": list of length n}
        """
        c = np.asarray(problem["c"], dtype=float)
        A = np.asarray(problem["A"], dtype=float)
        b = np.asarray(problem["b"], dtype=float)

        n = int(c.shape[0])

        # Handle case when A or b might be empty
        # Normalize A to shape (m, n), possibly with m=0
        if A.size == 0:
            # No inequality constraints (other than box)
            # Optimal solution under 0<=x<=1 is:
            #   x_i = 1 if c_i < 0, 0 if c_i > 0, any in [0,1] if c_i == 0
            # Choose x_i = 0 when c_i == 0 to be deterministic
            x = np.where(c < 0, 1.0, 0.0)
            return {"solution": x.tolist()}

        # Ensure A has shape (m, n)
        if A.ndim == 1:
            # Single constraint row
            A = A.reshape(1, -1)
        m = int(A.shape[0])

        # If there are actually zero constraints (m == 0), same shortcut
        if m == 0:
            x = np.where(c < 0, 1.0, 0.0)
            return {"solution": x.tolist()}

        # Use SciPy HiGHS if available
        if _HAS_SCIPY:
            bounds = [(0.0, 1.0)] * n
            A_ub = A
            b_ub = b
            # Attempt to solve using HiGHS
            res = linprog(
                c,
                A_ub=A_ub,
                b_ub=b_ub,
                bounds=bounds,
                method="highs",
            )
            if not res.success:
                # Fall back to a simple heuristic: project greedy solution onto constraints using clipping
                # Though this should almost never trigger, keep a safe fallback.
                x0 = np.where(c < 0, 1.0, 0.0)
                # Simple feasibility repair via gradient-like step (very lightweight)
                # Iterate a few times to reduce constraint violations
                for _ in range(20):
                    viol = A @ x0 - b
                    idx = np.where(viol > 0)[0]
                    if idx.size == 0:
                        break
                    # Move against weighted average of violated normals
                    step_dir = A[idx].sum(axis=0)
                    # If all zeros, break
                    if not np.any(step_dir):
                        break
                    # Move small step towards feasibility, respecting bounds
                    alpha = 0.1
                    x0 = np.clip(x0 - alpha * step_dir, 0.0, 1.0)
                return {"solution": x0.tolist()}
            return {"solution": res.x.tolist()}

        # If SciPy unavailable (unlikely), implement a very small projected subgradient descent as a fallback.
        # This is not as strong as HiGHS but should handle simple cases reasonably.
        x = np.where(c < 0, 1.0, 0.0)
        lr = 0.1
        for _ in range(200):
            # Subgradient for inequality violations
            viol = A @ x - b
            idx = viol > 0
            if not np.any(idx):
                break
            grad = c + A[idx].sum(axis=0)
            x = np.clip(x - lr * grad, 0.0, 1.0)
            lr *= 0.99  # small decay
        return {"solution": x.tolist()}