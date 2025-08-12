from typing import Any, Dict
import numpy as np

class Solver:
    def solve(self, problem: Dict[str, Any], **kwargs) -> Any:
        """
        Solve the power control LP:
            minimize sum(P)
            s.t.  P_min <= P <= P_max
                  for each i: G_ii * P_i >= S_min * (sigma_i + sum_{k!=i} G_ik * P_k)

        Returns:
            {"P": [...], "objective": float}
        """
        # Parse G
        if "G" not in problem:
            raise ValueError("Missing 'G' in problem")
        G = np.asarray(problem["G"], dtype=float)
        if G.ndim != 2 or G.shape[0] != G.shape[1]:
            raise ValueError("'G' must be a square matrix")
        n = G.shape[0]

        # Parse noise (support several key names, including Unicode sigma)
        sigma = None
        for key in ("σ", "sigma", "s", "noise"):
            if key in problem:
                sigma = np.asarray(problem[key], dtype=float).ravel()
                break
        if sigma is None:
            raise ValueError("Missing noise powers 'σ' in problem")
        if sigma.size == 1 and n > 1:
            sigma = np.full(n, float(sigma))
        if sigma.size != n:
            raise ValueError("Length of 'σ' must equal dimension of G")

        # Parse P_min
        if "P_min" not in problem:
            raise ValueError("Missing 'P_min' in problem")
        P_min = np.asarray(problem["P_min"], dtype=float).ravel()
        if P_min.size == 1 and n > 1:
            P_min = np.full(n, float(P_min))
        if P_min.size != n:
            raise ValueError("Length of 'P_min' must equal dimension of G")

        # Parse P_max (flexible shapes)
        if "P_max" not in problem:
            raise ValueError("Missing 'P_max' in problem")
        P_max_arr = np.asarray(problem["P_max"], dtype=float)
        if P_max_arr.size == 1 and n > 1:
            P_max = np.full(n, float(P_max_arr))
        else:
            P_max = P_max_arr.ravel()[:n]
            if P_max.size != n:
                raise ValueError("Length of 'P_max' must equal dimension of G")

        S_min = float(problem.get("S_min", 0.0))

        if (P_min > P_max).any():
            raise ValueError("Each entry of P_min must be <= corresponding entry of P_max")

        # No SINR requirement: minimal power is P_min (clipped)
        if S_min <= 0:
            P_opt = np.clip(P_min, P_min, P_max)
            return {"P": P_opt.tolist(), "objective": float(np.sum(P_opt))}

        # Build linear constraints A @ P >= rhs
        # A[i,i] = G_ii ; A[i,k!=i] = -S_min * G[i,k]
        A = -S_min * G.copy()
        diag = np.diag(G).copy()
        idx = np.arange(n)
        A[idx, idx] = diag
        rhs = S_min * sigma

        tol = 1e-9

        # If P_min already satisfies constraints, it's optimal (smallest feasible due to monotonicity)
        if np.all(A.dot(P_min) >= rhs - tol):
            P_opt = np.clip(P_min, P_min, P_max)
            return {"P": P_opt.tolist(), "objective": float(np.sum(P_opt))}

        # Try direct closed-form fixed-point solution: P = (I - F)^{-1} u
        # This is the minimal P satisfying the SINR constraints when no bounds are active.
        if np.all(diag > 0):
            F_try = S_min * (G / diag[:, None])
            F_try[idx, idx] = 0.0
            u_try = (S_min * sigma) / diag
            # Solve (I - F) P = u
            try:
                M = np.eye(n) - F_try
                P_unb = np.linalg.solve(M, u_try)
                if np.isfinite(P_unb).all():
                    # If unbounded solution within bounds, it's optimal
                    if np.all(P_unb >= P_min - 1e-9) and np.all(P_unb <= P_max + 1e-9) and np.all(A.dot(P_unb) >= rhs - 1e-8):
                        return {"P": P_unb.tolist(), "objective": float(np.sum(P_unb))}
            except np.linalg.LinAlgError:
                pass

        # Try SciPy linprog (HiGHS) for speed
        try:
            from scipy.optimize import linprog

            c = np.ones(n, dtype=float)
            A_ub = -A  # convert A @ P >= rhs to -A @ P <= -rhs
            b_ub = -rhs
            bounds = [(float(P_min[i]), float(P_max[i])) for i in range(n)]
            res = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method="highs")
            if res is not None and getattr(res, "success", False) and getattr(res, "x", None) is not None:
                P_sol = np.clip(res.x, P_min, P_max)
                if np.all(A.dot(P_sol) >= rhs - 1e-8):
                    return {"P": P_sol.tolist(), "objective": float(np.sum(P_sol))}
        except Exception:
            # SciPy not available or failed; fall through to other methods
            pass

        # Monotone fixed-point iteration fallback (requires positive diagonal)
        if np.all(diag > 0):
            u = (S_min * sigma) / diag
            F = S_min * (G / diag[:, None])
            F[idx, idx] = 0.0
            P = P_min.copy()
            max_iter = 5000 if n <= 200 else 2000
            for _ in range(max_iter):
                P_next = F.dot(P) + u
                # enforce bounds
                P_next = np.maximum(P_min, np.minimum(P_max, P_next))
                if np.allclose(P_next, P, atol=1e-12, rtol=0):
                    if np.all(A.dot(P_next) >= rhs - 1e-8):
                        return {"P": P_next.tolist(), "objective": float(np.sum(P_next))}
                    break
                P = P_next
            if np.all(A.dot(P) >= rhs - 1e-8) and np.all(P >= P_min - 1e-12) and np.all(P <= P_max + 1e-12):
                return {"P": P.tolist(), "objective": float(np.sum(P))}

        # Final fallback: CVXPY if available
        try:
            import cvxpy as cp

            P_var = cp.Variable(n)
            constraints = [P_var >= P_min, P_var <= P_max, A @ P_var >= rhs]
            prob = cp.Problem(cp.Minimize(cp.sum(P_var)), constraints)
            # Try ECOS then SCS
            prob.solve(solver=cp.ECOS, verbose=False)
            if prob.status not in (cp.OPTIMAL, cp.OPTIMAL_INACCURATE):
                prob.solve(solver=cp.SCS, verbose=False)
            if prob.status not in (cp.OPTIMAL, cp.OPTIMAL_INACCURATE):
                raise RuntimeError(f"CVXPY solver failed (status={prob.status})")
            Pval = np.asarray(P_var.value, dtype=float).ravel()
            Pval = np.clip(Pval, P_min, P_max)
            if np.all(A.dot(Pval) >= rhs - 1e-6):
                return {"P": Pval.tolist(), "objective": float(np.sum(Pval))}
            raise RuntimeError("CVXPY produced infeasible solution")
        except Exception as e:
            raise RuntimeError(f"Unable to solve problem with available methods: {e}")