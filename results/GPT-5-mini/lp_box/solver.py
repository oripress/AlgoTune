from typing import Any, Dict, List
import numpy as np

class Solver:
    def solve(self, problem: Dict[str, Any], **kwargs) -> Any:
        """
        Solve the LP box problem:
            minimize    c^T x
            subject to  A x <= b
                        0 <= x <= 1

        Strategy:
        - Compute trivial box solution x_i = 1 if c_i < 0 else 0. If it satisfies Ax <= b,
          return it (it's optimal).
        - Otherwise, call scipy.optimize.linprog with the HiGHS solver for a fast exact solution.
        - If linprog is unavailable or fails, run a lightweight greedy repair heuristic to
          produce a feasible solution starting from the box solution.
        """
        # Read input vectors/matrices
        c = np.asarray(problem.get("c", []), dtype=float)
        if c.size == 0:
            return {"solution": []}
        n = int(c.size)

        A_in = problem.get("A", [])
        b_in = problem.get("b", [])

        # Trivial box-only optimum: x_i = 1 if c_i < 0 else 0
        box_x = np.where(c < 0.0, 1.0, 0.0).astype(float)

        # If no linear constraints provided, box_x is optimal
        if A_in is None:
            return {"solution": box_x.tolist()}
        if isinstance(A_in, list) and len(A_in) == 0:
            return {"solution": box_x.tolist()}

        # Convert A and b to numpy arrays robustly
        try:
            A = np.asarray(A_in, dtype=float)
        except Exception:
            return {"solution": box_x.tolist()}

        if A.size == 0:
            return {"solution": box_x.tolist()}

        # Normalize A to shape (m, n)
        if A.ndim == 1:
            # If it matches n, treat as single row; else attempt reshape
            if A.shape[0] == n:
                A = A.reshape(1, n)
            else:
                try:
                    A = A.reshape(-1, n)
                except Exception:
                    return {"solution": box_x.tolist()}
        else:
            if A.shape[1] != n:
                # try transpose if that matches
                if A.shape[0] == n and A.shape[1] != n:
                    A = A.T
                else:
                    try:
                        A = A.reshape(-1, n)
                    except Exception:
                        return {"solution": box_x.tolist()}

        # Process b
        try:
            b = np.asarray(b_in, dtype=float).reshape(-1)
        except Exception:
            return {"solution": box_x.tolist()}

        m = A.shape[0]
        if b.size == 0:
            return {"solution": box_x.tolist()}

        if b.shape[0] != m:
            # If scalar, broadcast; otherwise attempt to broadcast
            if b.size == 1:
                b = np.full((m,), float(b))
            else:
                try:
                    b = np.broadcast_to(b, (m,))
                except Exception:
                    return {"solution": box_x.tolist()}

        # Quick check: if box solution satisfies A x <= b, return it (optimal)
        try:
            if np.all(A.dot(box_x) <= b + 1e-12):
                return {"solution": box_x.tolist()}
        except Exception:
            # If numerical issue, continue to solver
            pass

        # Try SciPy linprog with HiGHS solver
        try:
            from scipy.optimize import linprog

            bounds = [(0.0, 1.0)] * n
            res = linprog(c, A_ub=A, b_ub=b, bounds=bounds, method="highs")
            if res is not None and getattr(res, "success", False) and getattr(res, "x", None) is not None:
                x = np.clip(res.x, 0.0, 1.0)
                x = np.nan_to_num(x, nan=0.0, posinf=1.0, neginf=0.0)
                return {"solution": x.tolist()}

            # Try alternative HiGHS variants if available
            for method in ("highs-ds", "highs-ipm"):
                try:
                    res2 = linprog(c, A_ub=A, b_ub=b, bounds=bounds, method=method)
                    if res2 is not None and getattr(res2, "success", False) and getattr(res2, "x", None) is not None:
                        x = np.clip(res2.x, 0.0, 1.0)
                        x = np.nan_to_num(x, nan=0.0, posinf=1.0, neginf=0.0)
                        return {"solution": x.tolist()}
                except Exception:
                    continue
        except Exception:
            # SciPy not available or solver error; fall back to heuristic
            pass

        # Fallback greedy repair: start from box_x and reduce variables to satisfy Ax <= b.
        # This heuristic aims to keep low objective cost but may not be optimal.
        x = box_x.copy()
        try:
            Ax = A.dot(x)
            tol = 1e-9
            max_steps = max(2000, 20 * n)
            steps = 0
            while np.any(Ax > b + tol) and steps < max_steps:
                steps += 1
                viol = Ax - b
                # index of most violated constraint
                j = int(np.argmax(viol))
                row = A[j]
                # indices that can be decreased (positive coeff and x > 0)
                candidates = np.where((row > 1e-12) & (x > 1e-12))[0]
                if candidates.size == 0:
                    break
                # choose candidate to decrease: prefer those that least worsen objective
                # For minimization, decreasing xi increases objective by c_i * (-dx). So prefer largest c_i.
                # Use ratio of c to row to account for effectiveness on constraint.
                scores = c[candidates] / (row[candidates] + 1e-12)
                idx = int(candidates[np.argmax(scores)])
                excess = Ax[j] - b[j]
                reduce_amt = min(x[idx], excess / max(row[idx], 1e-12))
                if reduce_amt <= 0:
                    x[idx] = 0.0
                else:
                    x[idx] -= reduce_amt
                Ax = A.dot(x)
            x = np.clip(x, 0.0, 1.0)
            return {"solution": x.tolist()}
        except Exception:
            # give up and return box solution
            return {"solution": box_x.tolist()}