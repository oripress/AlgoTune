import numpy as np
import scipy.sparse as sp
import ecos

class Solver:
    def solve(self, problem, **kwargs):
        # Parse problem data
        c = np.asarray(problem["c"], dtype=float)
        b = np.asarray(problem["b"], dtype=float)
        P_py = problem["P"]
        q_py = problem["q"]
        m = len(b)
        n = len(c)

        # Cone dimensions: m second-order cones
        dims = {"l": 0, "q": [n + 1] * m, "s": []}

        # Assemble G and h
        if m > 0:
            G_blocks = []
            h_blocks = []
            for i in range(m):
                q_i = np.asarray(q_py[i], dtype=float).flatten()
                P_i = np.asarray(P_py[i], dtype=float)
                # Constraint: [b_i - q_i^T x; P_i^T x] in SOC
                G_i = np.vstack((q_i.reshape(1, n), -P_i.T))
                G_blocks.append(G_i)
                h_i = np.concatenate(([b[i]], np.zeros(n, dtype=float)))
                h_blocks.append(h_i)
            G = np.vstack(G_blocks)
            h = np.concatenate(h_blocks)
        else:
            G = np.zeros((0, n), dtype=float)
            h = np.zeros((0,), dtype=float)

        # Convert G to sparse format for ECOS
        G_sp = sp.csc_matrix(G)

        # Solve SOCP using ECOS
        try:
            sol = ecos.solve(c, G_sp, h, dims, verbose=False)
        except Exception:
            return {"objective_value": float("inf"), "x": np.full(n, np.nan)}

        # Extract solution x
        x = None
        if isinstance(sol, (list, tuple)) and len(sol) > 0:
            x = sol[0]
        else:
            try:
                x = sol['x']
            except Exception:
                pass

        if x is None:
            return {"objective_value": float("inf"), "x": np.full(n, np.nan)}

        x_arr = np.asarray(x, dtype=float).flatten()

        # Validate solution
        if x_arr.shape[0] != n or np.any(~np.isfinite(x_arr)):
            return {"objective_value": float("inf"), "x": np.full(n, np.nan)}

        # Compute objective value
        obj_val = float(np.dot(c, x_arr))
        return {"objective_value": obj_val, "x": x_arr}