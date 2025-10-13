from __future__ import annotations

from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
import scipy.sparse as sp

class Solver:
    def _to_array(self, x: Any, dtype=float) -> np.ndarray:
        return np.asarray(x, dtype=dtype)

    def solve(self, problem: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        """
        Solve the robust linear program via SOCP using a direct ECOS formulation:
            minimize    c^T x
            subject to  q_i^T x + ||P_i^T x||_2 <= b_i  for all i

        Input:
            problem: dict with keys
                - "c": list/array of shape (n,)
                - "b": list/array of shape (m,)
                - "P": list of m matrices (each shape (n, r_i), typically (n, n))
                - "q": list of m vectors (each shape (n,))

        Output:
            dict with:
                - "objective_value": float
                - "x": np.ndarray of shape (n,)
        """
        # Parse inputs
        c = self._to_array(problem["c"], dtype=float).reshape(-1)
        b = self._to_array(problem["b"], dtype=float).reshape(-1)

        P_in = problem["P"]
        q_in = problem["q"]

        # Convert P and q to lists of numpy arrays with correct shapes
        P_list: List[np.ndarray] = []
        q_list: List[np.ndarray] = []

        m = len(b)
        n = c.size

        if m == 0:
            # No constraints -> unbounded (return inf and NaNs to match reference behavior)
            return {"objective_value": float("inf"), "x": np.full(n, np.nan, dtype=float)}

        if not isinstance(P_in, (list, tuple)) or not isinstance(q_in, (list, tuple)):
            # Invalid structure, fall back to safe behavior
            return {"objective_value": float("inf"), "x": np.full(n, np.nan, dtype=float)}

        if len(P_in) != m or len(q_in) != m:
            # Dimension mismatch
            return {"objective_value": float("inf"), "x": np.full(n, np.nan, dtype=float)}

        for i in range(m):
            Pi = self._to_array(P_in[i], dtype=float)
            qi = self._to_array(q_in[i], dtype=float).reshape(-1)
            if qi.size != n:
                return {"objective_value": float("inf"), "x": np.full(n, np.nan, dtype=float)}
            # Ensure Pi has correct second dim equal to n (so that Pi.T @ x is valid)
            if Pi.ndim != 2 or Pi.shape[0] != n:
                # Reference expects P_i.T @ x, which requires Pi.shape[0] == n
                return {"objective_value": float("inf"), "x": np.full(n, np.nan, dtype=float)}

            P_list.append(Pi)
            q_list.append(qi)

        # Build ECOS SOCP data:
        # Minimize c^T x
        # For each i, SOC constraint [ t ; u ] in SOC with
        #   t = b_i - q_i^T x
        #   u = P_i^T x
        # This is encoded as s = h - G x in SOC: set
        #   h_i = [ b_i ; 0 ]
        #   G_i = [ q_i^T ; -P_i^T ]
        # and dims['q'] += [1 + r_i], where r_i = number of columns of P_i (dim of u)
        try:
            import ecos  # Local import to avoid import cost if not used elsewhere

            # Stack G and h blocks
            G_blocks = []
            h_blocks = []
            q_dims: List[int] = []

            for i in range(m):
                Pi = P_list[i]
                qi = q_list[i]
                bi = float(b[i])

                # u dimension is the number of columns of Pi
                ri = Pi.shape[1]
                q_dims.append(1 + ri)

                # Construct block:
                # Top row: qi^T (shape 1 x n)
                G_top = sp.csc_matrix(qi.reshape(1, -1))
                # Bottom rows: -Pi^T (shape ri x n)
                G_bottom = sp.csc_matrix(-Pi.T)

                # Stack vertically
                G_i = sp.vstack([G_top, G_bottom], format="csc")

                # h block: [b_i; zeros(ri)]
                h_i = np.zeros(1 + ri, dtype=float)
                h_i[0] = bi

                G_blocks.append(G_i)
                h_blocks.append(h_i)

            if len(G_blocks) == 1:
                G = G_blocks[0]
                h_vec = h_blocks[0]
            else:
                G = sp.vstack(G_blocks, format="csc")
                h_vec = np.concatenate(h_blocks, axis=0)

            dims = {"l": 0, "q": q_dims, "e": 0}
            A = None
            b_eq = None

            # Solve with ECOS
            sol = ecos.solve(c=c, G=G, h=h_vec, dims=dims, A=A, b=b_eq, **kwargs)
            x_opt = sol.get("x", None)
            info = sol.get("info", {})

            # ECOS exitFlag == 0 indicates success
            if x_opt is None or info.get("exitFlag", None) != 0:
                # Fall back to cvxpy if ECOS did not find an optimal solution
                return self._fallback_cvxpy(P_list, q_list, b, c)

            x_opt = np.asarray(x_opt, dtype=float).reshape(-1)
            if x_opt.size != n or not np.all(np.isfinite(x_opt)):
                return self._fallback_cvxpy(P_list, q_list, b, c)

            obj = float(np.dot(c, x_opt))
            if not np.isfinite(obj):
                return self._fallback_cvxpy(P_list, q_list, b, c)

            return {"objective_value": obj, "x": x_opt}

        except Exception:
            # Any error: fallback to cvxpy
            return self._fallback_cvxpy(P_list, q_list, b, c)

    def _fallback_cvxpy(
        self, P_list: List[np.ndarray], q_list: List[np.ndarray], b: np.ndarray, c: np.ndarray
    ) -> Dict[str, Any]:
        """Fallback to CVXPY solution if ECOS fails."""
        try:
            import cvxpy as cp

            n = c.size
            m = len(P_list)

            x = cp.Variable(n)
            constraints = []
            for i in range(m):
                Pi = P_list[i]
                qi = q_list[i]
                bi = float(b[i])
                constraints.append(cp.SOC(bi - qi @ x, Pi.T @ x))

            prob = cp.Problem(cp.Minimize(c @ x), constraints)

            # Try ECOS first; if fails, try CLARABEL; then default
            try:
                prob.solve(solver=cp.ECOS, verbose=False)
            except Exception:
                try:
                    prob.solve(solver=cp.CLARABEL, verbose=False)
                except Exception:
                    prob.solve(verbose=False)

            if prob.status not in ["optimal", "optimal_inaccurate"]:
                return {"objective_value": float("inf"), "x": np.full(n, np.nan, dtype=float)}

            x_val = np.asarray(x.value, dtype=float).reshape(-1)
            obj = float(np.dot(c, x_val))
            return {"objective_value": obj, "x": x_val}
        except Exception:
            # If even fallback fails, declare infeasible/unbounded to match reference behavior
            n = c.size
            return {"objective_value": float("inf"), "x": np.full(n, np.nan, dtype=float)}