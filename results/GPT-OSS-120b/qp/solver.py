from typing import Any, Dict
import numpy as np
import cvxpy as cp
import scipy.sparse as sp

# Try to import the native OSQP solver; if unavailable we will fall back to CVXPY.
try:
    import osqp
    _OSQP_AVAILABLE = True
except Exception:  # pragma: no cover
    _OSQP_AVAILABLE = False

class Solver:
    def solve(self, problem: Dict[str, Any]) -> Dict[str, Any]:
        """
        Solve a convex quadratic program defined by:
            minimize (1/2) x^T P x + q^T x
            subject to G x <= h, A x == b

        Parameters
        ----------
        problem : dict
            Dictionary with keys "P"/"Q", "q", "G", "h", "A", "b".

        Returns
        -------
        dict
            {"solution": [...]}  optimal primal vector.
        """
        # ------------------------------------------------------------------
        # 1. Load data (support both "P" and legacy "Q")
        # ------------------------------------------------------------------
        P = np.asarray(problem.get("P", problem.get("Q", [])), dtype=float)
        q = np.asarray(problem.get("q", []), dtype=float)
        G = np.asarray(problem.get("G", []), dtype=float)
        h = np.asarray(problem.get("h", []), dtype=float)
        A = np.asarray(problem.get("A", []), dtype=float)
        b = np.asarray(problem.get("b", []), dtype=float)

        # Determine dimension
        n = P.shape[0] if P.size else q.shape[0]

        # Ensure symmetry of P
        if P.size:
            P = (P + P.T) / 2

        # ------------------------------------------------------------------
        # 2. Simple case: no constraints -> solve linear system
        # ------------------------------------------------------------------
        if G.size == 0 and h.size == 0 and A.size == 0:
            # Solve 0.5 x^T P x + q^T x  =>  P x = -q  (if P is PSD, use pseudo‑inverse)
            if P.size:
                # Use pseudo‑inverse for singular P
                x = -np.linalg.pinv(P) @ q
            else:
                # Pure linear objective: any x works; choose zero vector
                x = np.zeros(n)
            return {"solution": x.tolist()}

        # ------------------------------------------------------------------
        # 3. Build OSQP problem if possible
        # ------------------------------------------------------------------
        if _OSQP_AVAILABLE:
            # Build combined constraint matrix
            # Inequalities: G x <= h  ->  l = -inf, u = h
            # Equalities: A x = b   ->  l = u = b
            # Stack them vertically
            if G.size and h.size:
                G_mat = G
                l_ineq = -np.inf * np.ones(G.shape[0])
                u_ineq = h
            else:
                G_mat = np.empty((0, n))
                l_ineq = np.empty(0)
                u_ineq = np.empty(0)

            if A.size and b.size:
                A_mat = A
                l_eq = b
                u_eq = b
            else:
                A_mat = np.empty((0, n))
                l_eq = np.empty(0)
                u_eq = np.empty(0)

            # Combine
            if G_mat.shape[0] + A_mat.shape[0] > 0:
                C = np.vstack([G_mat, A_mat])
                l = np.concatenate([l_ineq, l_eq])
                u = np.concatenate([u_ineq, u_eq])
                # Convert to sparse format required by OSQP
                C_sp = sp.csc_matrix(C)
                # OSQP expects P to be sparse and upper‑triangular
                P_sp = sp.csc_matrix(P)
                # OSQP requires P to be positive semidefinite; we already symmetrized
                prob = osqp.OSQP()
                prob.setup(P=P_sp, q=q, A=C_sp, l=l, u=u, verbose=False, eps_abs=1e-8, eps_rel=1e-8)
                res = prob.solve()
                if res.info.status_val not in (osqp.constant('OSQP_SOLVED'), osqp.constant('OSQP_SOLVED_INACCURATE')):
                    # Fallback to CVXPY if OSQP fails
                    _use_cvx = True
                else:
                    x_opt = res.x
                    return {"solution": x_opt.tolist()}
            else:
                # No constraints after all (should have been caught earlier)
                _use_cvx = True
        else:
            _use_cvx = True

        # ------------------------------------------------------------------
        # 4. Fallback to CVXPY (OSQP via CVXPY) if needed
        # ------------------------------------------------------------------
        # Define CVXPY variable
        x = cp.Variable(n)
        if P.size:
            objective = 0.5 * cp.quad_form(x, cp.psd_wrap(P)) + q @ x
        else:
            objective = q @ x

        constraints = []
        if G.size and h.size:
            constraints.append(G @ x <= h)
        if A.size and b.size:
            constraints.append(A @ x == b)

        prob = cp.Problem(cp.Minimize(objective), constraints)
        try:
            prob.solve(solver=cp.OSQP, eps_abs=1e-8, eps_rel=1e-8)
        except Exception:
            prob.solve()
        if prob.status not in (cp.OPTIMAL, cp.OPTIMAL_INACCURATE):
            raise ValueError(f"Quadratic program solver failed with status {prob.status}")
        return {"solution": x.value.tolist()}