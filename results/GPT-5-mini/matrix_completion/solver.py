from typing import Any, Dict, List, Tuple

import numpy as np
import cvxpy as cp

class Solver:
    def solve(self, problem: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        """
        Solve the Perron-Frobenius matrix completion using a log-domain convex formulation.

        Variables:
            b_k = log(B_ij) for missing entries
            s_i = log(x_i) scaling vector (we fix s_0 = 0 for stability)
            tau = log(lambda) where lambda is the Perron root

        Constraints (for each i):
            log_sum_exp_j( log(B_ij) + s_j - s_i ) <= tau

        Product constraint for missing entries:
            sum_k b_k == 0  (ensures product of missing B_ij = 1)
        """
        inds = np.array(problem.get("inds", []))
        a = np.array(problem.get("a", []), dtype=float)
        n = int(problem["n"])

        # Normalize inds to shape (k,2)
        if inds.size == 0:
            inds2 = np.zeros((0, 2), dtype=int)
        else:
            inds2 = np.array(inds, dtype=int)
            if inds2.ndim == 1:
                inds2 = inds2.reshape(1, 2)

        # Observed mask and log-values for observed entries
        observed_mask = np.zeros((n, n), dtype=bool)
        logA = np.zeros((n, n), dtype=float)
        if inds2.shape[0] > 0:
            for idx in range(inds2.shape[0]):
                i, j = int(inds2[idx, 0]), int(inds2[idx, 1])
                observed_mask[i, j] = True
                val = float(a[idx])
                if val <= 0:
                    # safeguard against nonpositive inputs (problem states positive entries)
                    val = max(val, 1e-300)
                logA[i, j] = float(np.log(val))

        # Identify missing coordinates and mapping to b indices
        missing_coords: List[Tuple[int, int]] = []
        map_missing = {}
        for i in range(n):
            for j in range(n):
                if not observed_mask[i, j]:
                    map_missing[(i, j)] = len(missing_coords)
                    missing_coords.append((i, j))
        m = len(missing_coords)

        # If no missing entries, return the given matrix and its Perron root
        if m == 0:
            B = np.zeros((n, n), dtype=float)
            if inds2.shape[0] > 0:
                for idx in range(inds2.shape[0]):
                    i, j = int(inds2[idx, 0]), int(inds2[idx, 1])
                    B[i, j] = float(a[idx])
            try:
                eigs = np.linalg.eigvals(B)
                rho = float(np.max(np.abs(eigs)))
            except Exception:
                rho = float(np.max(np.abs(B)))
            return {"B": B.tolist(), "optimal_value": float(rho)}

        # CVXPY variables: b (logs of missing entries), s (logs of scaling), tau (log lambda)
        b = cp.Variable(m)
        s = cp.Variable(n)
        tau = cp.Variable()

        constraints = []
        # Fix one component of s to remove translation invariance (improves conditioning)
        constraints.append(s[0] == 0)

        # Build per-row log-sum-exp constraints
        for i in range(n):
            terms = []
            for j in range(n):
                if observed_mask[i, j]:
                    terms.append(logA[i, j] + s[j] - s[i])
                else:
                    k = map_missing[(i, j)]
                    terms.append(b[k] + s[j] - s[i])
            # log_sum_exp(...) <= tau
            # cp.log_sum_exp accepts a vector expression; hstack the scalar terms
            constraints.append(cp.log_sum_exp(cp.hstack(terms)) <= tau)

        # product constraint in log domain (sum of logs = 0)
        constraints.append(cp.sum(b) == 0)

        prob = cp.Problem(cp.Minimize(tau), constraints)

        # Try ECOS first (interior-point), then SCS; fall back to GP if necessary.
        solved = False
        try:
            _ = prob.solve(solver=cp.ECOS, verbose=False)
            solved = True
        except Exception:
            try:
                _ = prob.solve(solver=cp.SCS, verbose=False, eps=1e-6, max_iters=10000)
                solved = True
            except Exception:
                solved = False

        if not solved or b.value is None or tau.value is None:
            # Fallback to the GP formulation using pf_eigenvalue (reference approach)
            try:
                Bvar = cp.Variable((n, n), pos=True)
                constraints2 = []
                # list all indices and find missing indices relative to observed ones
                xx, yy = np.meshgrid(np.arange(n), np.arange(n))
                allinds = np.vstack((yy.flatten(), xx.flatten())).T
                if inds2.shape[0] == 0:
                    otherinds = allinds
                else:
                    otherinds = allinds[~(allinds == inds2[:, None]).all(2).any(0)]
                if otherinds.shape[0] > 0:
                    constraints2.append(cp.prod(Bvar[otherinds[:, 0], otherinds[:, 1]]) == 1.0)
                if inds2.shape[0] > 0:
                    constraints2.append(Bvar[inds2[:, 0], inds2[:, 1]] == a)
                obj2 = cp.Minimize(cp.pf_eigenvalue(Bvar))
                prob2 = cp.Problem(obj2, constraints2)
                res2 = prob2.solve(gp=True)
                if Bvar.value is None:
                    raise RuntimeError("GP fallback failed to produce a value")
                return {"B": Bvar.value.tolist(), "optimal_value": float(res2)}
            except Exception:
                # Ultimate fallback: fill missing with 1 and compute PF root
                B_fallback = np.ones((n, n), dtype=float)
                if inds2.shape[0] > 0:
                    for idx in range(inds2.shape[0]):
                        i, j = int(inds2[idx, 0]), int(inds2[idx, 1])
                        B_fallback[i, j] = float(a[idx])
                try:
                    eigs = np.linalg.eigvals(B_fallback)
                    rho_fb = float(np.max(np.abs(eigs)))
                except Exception:
                    rho_fb = float(np.max(np.abs(B_fallback)))
                return {"B": B_fallback.tolist(), "optimal_value": float(rho_fb)}

        # Construct B from solution: observed entries from 'a', missing entries = exp(b)
        B = np.zeros((n, n), dtype=float)
        if inds2.shape[0] > 0:
            for idx in range(inds2.shape[0]):
                i, j = int(inds2[idx, 0]), int(inds2[idx, 1])
                B[i, j] = float(a[idx])
        b_val = np.asarray(b.value).flatten()
        for k, (i, j) in enumerate(missing_coords):
            bk = float(b_val[k]) if np.isfinite(b_val[k]) else 0.0
            bk = float(np.clip(bk, -700.0, 700.0))
            B[i, j] = float(np.exp(bk))

        # Compute accurate Perron root using numpy eigenvalues
        try:
            eigs = np.linalg.eigvals(B)
            rho = float(np.max(np.abs(eigs)))
        except Exception:
            try:
                rho = float(np.exp(float(tau.value))) if tau.value is not None else float(np.max(np.abs(B)))
            except Exception:
                rho = float(np.max(np.abs(B)))

        return {"B": B.tolist(), "optimal_value": float(rho)}