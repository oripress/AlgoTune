import numpy as np
import osqp
from scipy import sparse

class Solver:
    def solve(self, problem: dict, **kwargs) -> dict:
        # Parse data (supporting keys "P" or "Q")
        P = np.asarray(problem.get("P") or problem.get("Q"), dtype=float)
        # Ensure symmetry
        P = (P + P.T) * 0.5
        q = np.asarray(problem["q"], dtype=float)
        # Handle possible missing constraints
        n = P.shape[0]
        G = np.asarray(problem.get("G") or [], dtype=float).reshape(-1, n)
        h = np.asarray(problem.get("h") or [], dtype=float).ravel()
        A = np.asarray(problem.get("A") or [], dtype=float).reshape(-1, n)
        b = np.asarray(problem.get("b") or [], dtype=float).ravel()
        # Count constraints
        m = G.shape[0]
        p = A.shape[0]

        # Unconstrained QP: x = -P^{-1} q
        if m == 0 and p == 0:
            try:
                x = -np.linalg.solve(P, q)
            except np.linalg.LinAlgError:
                pass
            else:
                return {"solution": x.tolist()}

        # Only equality constraints: solve KKT system
        if m == 0 and p > 0:
            KKT = np.block([[P, A.T], [A, np.zeros((p, p), dtype=float)]])
            rhs = np.hstack([-q, b])
            try:
                sol = np.linalg.solve(KKT, rhs)
            except np.linalg.LinAlgError:
                pass
            else:
                return {"solution": sol[:n].tolist()}
        # Convert to sparse
        P_csc = sparse.csc_matrix(P)

        # Build constraint stacks
        A_blocks = []
        l_blocks = []
        u_blocks = []
        # Inequalities G x <= h  -->  -inf <= Gx <= h
        if G.size:
            A_blocks.append(sparse.csc_matrix(G))
            l_blocks.append(-np.inf * np.ones(h.shape, dtype=float))
            u_blocks.append(h)
        # Equalities A x == b
        if A.size:
            A_blocks.append(sparse.csc_matrix(A))
            l_blocks.append(b)
            u_blocks.append(b)
        # Stack or empty
        if A_blocks:
            A_all = sparse.vstack(A_blocks, format="csc")
            l_all = np.hstack(l_blocks)
            u_all = np.hstack(u_blocks)
        else:
            A_all = sparse.csc_matrix((0, n))
            l_all = np.empty(0, dtype=float)
            u_all = np.empty(0, dtype=float)
        # Two-phase OSQP solve: quick then fallback
        prob = osqp.OSQP()
        prob.setup(P=P_csc, q=q, A=A_all, l=l_all, u=u_all,
                   eps_abs=1e-6, eps_rel=1e-6, max_iter=200,
                   polish=False, verbose=False)
        res = prob.solve()
        x = res.x
        # Check feasibility
        if x is not None and (not G.size or np.all(G.dot(x) - h <= 1e-6)) and (not A.size or np.allclose(A.dot(x), b, atol=1e-6)):
            return {"solution": x.tolist()}
        # Fallback: refine with polish
        prob.update_settings(polish=True)
        res = prob.solve()
        return {"solution": res.x.tolist()}