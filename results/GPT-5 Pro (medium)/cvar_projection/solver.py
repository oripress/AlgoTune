from __future__ import annotations

from typing import Any, Dict, List, Tuple

import numpy as np


def _topk_indices(arr: np.ndarray, k: int) -> np.ndarray:
    """Return indices of the k largest elements of arr (ties broken arbitrarily)."""
    if k <= 0:
        return np.empty((0,), dtype=int)
    if k >= arr.size:
        return np.argsort(arr)
    # Use argpartition for efficiency, then sort those k by value descending
    idx = np.argpartition(arr, -k)[-k:]
    # Sort selected indices by their values (descending) for determinism
    idx_sorted = idx[np.argsort(arr[idx], kind="mergesort")]
    return idx_sorted


class Solver:
    def solve(self, problem: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        """
        Compute the projection onto the CVaR constraint set defined by sum of top-k losses.

        minimize ||x - x0||^2
        subject to sum of k largest entries of A x <= alpha, where alpha = kappa * k and k = int((1-beta)*m)

        We implement an outer-approximation with active halfspaces:
          - The CVaR constraint equals the intersection over all subsets S of size k of halfspaces:
                (sum_{i in S} A_i)^T x <= alpha.
          - We iteratively identify violated top-k sets at the current iterate and add their halfspace.
          - Projection onto the intersection of gathered halfspaces is solved exactly via an active-set method
            on the dual (equivalently projecting onto a polyhedron defined by halfspaces).

        :param problem: dict with keys "x0", "loss_scenarios", "beta", "kappa"
        :return: dict with key "x_proj"
        """
        # Extract data
        try:
            x0 = np.asarray(problem["x0"], dtype=float).ravel()
            A = np.asarray(problem["loss_scenarios"], dtype=float)
        except Exception:
            return {"x_proj": []}

        if A.ndim != 2:
            return {"x_proj": []}
        m, n = A.shape
        if x0.size != n:
            return {"x_proj": []}

        beta = float(problem.get("beta", 0.95))
        kappa = float(problem.get("kappa", 0.0))

        # Compute k and alpha as in reference
        k_raw = int((1.0 - beta) * m)
        # Handle degenerate k
        if k_raw <= 0:
            # No effective CVaR constraint; projection is identity
            return {"x_proj": x0.tolist()}

        k = k_raw
        alpha = kappa * k

        # Helper: compute sum of top-k of A @ x
        def sum_topk(Ax: np.ndarray) -> float:
            if k >= m:
                return float(np.sum(Ax))
            idx = np.argpartition(Ax, -k)[-k:]
            return float(np.sum(Ax[idx]))

        # Quick feasibility check: if x0 already feasible, return it
        y0 = A @ x0
        if sum_topk(y0) <= alpha + 1e-10:
            return {"x_proj": x0.tolist()}

        # Store constraints as sets S and their normals p_S = sum_{i in S} A_i
        normals: List[np.ndarray] = []
        norms2: List[float] = []
        seen_S: set[Tuple[int, ...]] = set()

        # Build initial violated top-k set from x0
        S_idx = _topk_indices(y0, k)
        # Ensure exactly k indices
        if S_idx.size != k:
            # Should not happen but safe-guard
            S_idx = np.argsort(y0)[-k:]
        S = tuple(sorted(int(i) for i in S_idx))
        if S not in seen_S:
            p = np.sum(A[np.array(S, dtype=int), :], axis=0)
            norm2 = float(np.dot(p, p))
            if norm2 > 0.0:
                normals.append(p)
                norms2.append(norm2)
                seen_S.add(S)

        # Projection onto intersection of halfspaces p_j^T x <= alpha for collected normals
        def project_halfspace_intersection(x_ref: np.ndarray) -> np.ndarray:
            """
            Solve min ||x - x_ref||^2 s.t. p_j^T x <= alpha for j=1..r exactly via active-set in dual space.
            """
            r = len(normals)
            if r == 0:
                return x_ref.copy()
            # Precompute Gram matrix G and c vector
            P = np.stack(normals, axis=1)  # n x r
            G = P.T @ P  # r x r
            c = P.T @ x_ref - alpha  # r-vector, c_j = p_j^T x_ref - alpha
            # If no violations at all, return x_ref
            if np.max(c) <= 1e-12:
                return x_ref.copy()

            # Active-set of constraints to enforce equality: indices into [0..r-1]
            # Initialize with the most violated constraint
            Aset: List[int] = [int(np.argmax(c))]
            # Solve iteratively
            max_inner_iters = 50 + 10 * r
            for _ in range(max_inner_iters):
                # Solve G_AA lambda_A = c_A for lambda_A
                A_idx = np.array(Aset, dtype=int)
                G_AA = G[np.ix_(A_idx, A_idx)]
                c_A = c[A_idx]
                # Use least-squares/pseudoinverse to be robust to rank deficiencies
                try:
                    # Try solve; if singular, fallback to lstsq
                    lambda_A = np.linalg.solve(G_AA, c_A)
                except np.linalg.LinAlgError:
                    lambda_A, *_ = np.linalg.lstsq(G_AA, c_A, rcond=None)

                # Enforce nonnegativity; if negative lambdas, drop the most negative and repeat
                if np.any(lambda_A < -1e-12):
                    # Drop the index with the most negative lambda
                    drop_idx = int(np.argmin(lambda_A))
                    del Aset[drop_idx]
                    if not Aset:
                        # If all dropped, pick next most violated and continue
                        viol = c.copy()
                        viol[viol < 0] = 0.0
                        if np.max(viol) <= 1e-12:
                            return x_ref.copy()
                        Aset = [int(np.argmax(viol))]
                    continue

                # Compute projected x
                x_proj = x_ref - P[:, A_idx] @ lambda_A

                # Check all constraints; compute v = p_j^T x - alpha = c_j - (G lambda)_j
                v = P.T @ x_proj - alpha  # equals p_j^T x_proj - alpha
                # Active constraints in Aset should be ~0 within tolerance
                # For others, allow <= 0; if violation positive, add worst violator
                # Tolerance
                tol = 1e-10
                v_max = float(np.max(v))
                if v_max <= tol:
                    # All constraints satisfied; done
                    return x_proj
                # Add the most violated constraint not already in Aset
                j_new = int(np.argmax(v))
                if j_new not in Aset:
                    Aset.append(j_new)
                    continue
                else:
                    # If the most violated is already active (due to numerical issues), perturb by removing any near-zero negative lambda
                    # Or break to avoid infinite loop
                    return x_proj

            # Fallback to Dykstra if active-set did not converge (rare)
            x = x_ref.copy()
            r = len(normals)
            cor = [np.zeros_like(x) for _ in range(r)]
            for _ in range(200):
                x_prev = x.copy()
                for j in range(r):
                    y = x + cor[j]
                    p = normals[j]
                    v = float(np.dot(p, y) - alpha)
                    if v > 0.0:
                        x_new = y - (v / norms2[j]) * p
                    else:
                        x_new = y
                    cor[j] = y - x_new
                    x = x_new
                if np.linalg.norm(x - x_prev) <= 1e-10 * (1.0 + np.linalg.norm(x_prev)):
                    break
            return x

        # Outer loop: add violated top-k halfspaces until feasible
        x = x0.copy()
        max_outer_iters = 50  # should be enough
        for _ in range(max_outer_iters):
            # Project onto intersection of current halfspaces
            x = project_halfspace_intersection(x0)

            y = A @ x
            # Check feasibility
            current_sum_topk = sum_topk(y)
            if current_sum_topk <= alpha + 1e-9:
                return {"x_proj": x.tolist()}

            # Identify new violated top-k set
            S_idx = _topk_indices(y, k)
            if S_idx.size != k:
                S_idx = np.argsort(y)[-k:]
            S = tuple(sorted(int(i) for i in S_idx))
            if S in seen_S:
                # Already included; if still infeasible due to numerical tolerance, slightly relax via single projection onto this halfspace
                p = np.sum(A[np.array(S, dtype=int), :], axis=0)
                norm2 = float(np.dot(p, p))
                if norm2 > 0:
                    v = float(np.dot(p, x) - alpha)
                    if v > 0:
                        x = x - (v / norm2) * p
                        # Check again
                        y = A @ x
                        if sum_topk(y) <= alpha + 1e-8:
                            return {"x_proj": x.tolist()}
                # If nothing changes, break to avoid infinite loop
                break
            # Add new constraint
            p_new = np.sum(A[np.array(S, dtype=int), :], axis=0)
            norm2_new = float(np.dot(p_new, p_new))
            if norm2_new <= 0.0:
                # Degenerate constraint; skip
                seen_S.add(S)
                continue
            normals.append(p_new)
            norms2.append(norm2_new)
            seen_S.add(S)

        # Final safeguard: if still not feasible due to numerical issues, perform a final Dykstra projection with all gathered constraints and return
        x = x0.copy()
        r = len(normals)
        cor = [np.zeros_like(x) for _ in range(r)]
        for _ in range(500):
            x_prev = x.copy()
            for j in range(r):
                y = x + cor[j]
                p = normals[j]
                v = float(np.dot(p, y) - alpha)
                if v > 0.0:
                    x_new = y - (v / norms2[j]) * p
                else:
                    x_new = y
                cor[j] = y - x_new
                x = x_new
            if np.linalg.norm(x - x_prev) <= 1e-10 * (1.0 + np.linalg.norm(x_prev)):
                break

        # Ensure feasibility mildly within tolerance; if still infeasible, return empty per reference behavior
        y = A @ x
        if sum_topk(y) > alpha + 1e-4 * max(1.0, abs(alpha)):
            return {"x_proj": []}

        return {"x_proj": x.tolist()}