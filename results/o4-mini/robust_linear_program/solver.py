import numpy as np
import ecos
import scipy.sparse as sp
class Solver:
    def solve(self, problem, **kwargs):
        # Parse inputs
        c = np.asarray(problem.get("c", []), dtype=float).ravel()
        b = np.asarray(problem.get("b", []), dtype=float).ravel()
        P = np.asarray(problem.get("P", []), dtype=float)
        q = np.asarray(problem.get("q", []), dtype=float)
        m = b.size
        n = c.size
        # Handle no constraints (unconstrained minimization)
        if m == 0:
            x0 = np.zeros(n, dtype=float)
            return {"objective_value": float(c.dot(x0)), "x": x0}
        # Build G and h for SOCP: ||P_i^T x|| <= b_i - q_i^T x
        rows = (n + 1) * m
        G = np.zeros((rows, n), dtype=float)
        h = np.zeros(rows, dtype=float)
        for i in range(m):
            off = i * (n + 1)
            # q_i^T x + s0 = b_i
            G[off, :] = q[i]
            h[off] = b[i]
            # -P_i^T x + s[1:] = 0
            G[off + 1 : off + 1 + n, :] = -P[i].T
        dims = {"l": 0, "q": [n + 1] * m}
        # Convert G to sparse CSC matrix for ECOS
        G = sp.csc_matrix(G)
        # Use relaxed tolerances and limited iterations for speed
        try:
            sol = ecos.solve(c, G, h, dims, verbose=False,
                             maxit=50, reltol=1e-3, feastol=1e-3, abstol=1e-3)
        except TypeError:
            sol = ecos.solve(c, G, h, dims, verbose=False)
        if sol is None:
            return {"objective_value": float("inf"), "x": np.full(n, np.nan)}
        # Unpack solution
        if isinstance(sol, dict):
            x = sol.get("x", None)
            info = sol.get("info", sol)
        else:
            x, _, _, info = sol
        if x is None:
            return {"objective_value": float("inf"), "x": np.full(n, np.nan)}
        x = np.asarray(x, dtype=float).ravel()[:n]
        # Check ECOS exit status
        exitflag = None
        if isinstance(info, dict):
            exitflag = info.get("exitflag", info.get("ExitFlag", info.get("exit_flag")))
        if exitflag is not None and exitflag != 0:
            return {"objective_value": float("inf"), "x": np.full(n, np.nan)}
        # Compute objective
        obj = float(c.dot(x))
        return {"objective_value": obj, "x": x}