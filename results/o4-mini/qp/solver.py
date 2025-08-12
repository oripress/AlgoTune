import numpy as np
import scipy.sparse as sp
import osqp

class Solver:
    def solve(self, problem, **kwargs):
        # support Q alias for P
        if "P" not in problem and "Q" in problem:
            problem["P"] = problem["Q"]
        # Extract problem data
        P = np.asarray(problem.get("P", problem.get("Q")), dtype=float)
        q = np.asarray(problem["q"], dtype=float).flatten()
        G = np.asarray(problem["G"], dtype=float)
        h = np.asarray(problem["h"], dtype=float).flatten()
        A = np.asarray(problem["A"], dtype=float)
        b = np.asarray(problem["b"], dtype=float).flatten()
        n = P.shape[0]
        # Reshape G to (m, n)
        if G.ndim == 1:
            if G.size == n:
                G = G.reshape(1, n)
            else:
                G = G.reshape(0, n)
        # Reshape A to (p, n)
        if A.ndim == 1:
            if A.size == n:
                A = A.reshape(1, n)
            else:
                A = A.reshape(0, n)
        m = G.shape[0]
        p = A.shape[0]
        # Direct solve for unconstrained or equality-only QPs
        if m == 0:
            try:
                if p == 0:
                    # Unconstrained QP: x = -P^{-1} q
                    x = -np.linalg.solve(P, q)
                else:
                    # Equality-only QP: solve KKT system
                    KKT = np.zeros((n + p, n + p), dtype=float)
                    KKT[:n, :n] = P
                    KKT[:n, n:] = A.T
                    KKT[n:, :n] = A
                    rhs = np.hstack([-q, b])
                    sol = np.linalg.solve(KKT, rhs)
                    x = sol[:n]
                return {"solution": x.tolist()}
            except np.linalg.LinAlgError:
                pass
        # Ensure P is symmetric
        P = (P + P.T) / 2
        # Build OSQP problem
        P_csc = sp.csc_matrix(P)
        G_csc = sp.csc_matrix(G) if m > 0 else sp.csc_matrix((0, n))
        A_csc = sp.csc_matrix(A) if p > 0 else sp.csc_matrix((0, n))
        A_stack = sp.vstack([G_csc, A_csc], format="csc")
        # Define bounds l <= Ax <= u
        inf = np.inf
        l = np.hstack([-inf * np.ones(m), b])
        u = np.hstack([h, b])
        solver = osqp.OSQP()
        solver.setup(P=P_csc, q=q, A=A_stack, l=l, u=u,
                     eps_abs=1e-4, eps_rel=1e-4, polish=True, verbose=False)
        res = solver.solve()
        x = res.x
        return {"solution": x.tolist()}