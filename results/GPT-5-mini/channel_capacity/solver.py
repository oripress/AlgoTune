import math
from typing import Any, Dict, Optional

import numpy as np

class Solver:
    def solve(self, problem: dict, **kwargs) -> Optional[Dict[str, Any]]:
        """
        Compute the channel capacity and optimal input distribution using the
        Blahut–Arimoto algorithm.

        Input:
            problem: dict with key "P" -> list of m lists of n floats (m x n matrix)
                   P_ij = P(Y=i | X=j)

        Output:
            dict with keys:
              - "x": list of n floats representing the optimal input distribution x.
              - "C": float representing the channel capacity in bits.
            Or None for invalid inputs.
        """
        if problem is None:
            return None

        P_list = problem.get("P")
        if P_list is None:
            return None

        P = np.asarray(P_list, dtype=float)
        if P.ndim != 2:
            return None

        m, n = P.shape
        if m <= 0 or n <= 0:
            return None

        # Basic validation
        if not np.all(np.isfinite(P)):
            return None
        # Allow tiny negative noise due to numerical input errors, but reject significant negatives
        if np.any(P < -1e-12):
            return None
        P = np.clip(P, 0.0, None)

        # Columns correspond to inputs; they must sum to 1
        if not np.allclose(np.sum(P, axis=0), 1.0, atol=1e-6):
            return None

        # Single-input channel has zero capacity
        if n == 1:
            return {"x": [1.0], "C": 0.0}

        # Precompute log(P) where P > 0
        maskP = P > 0.0
        logP = np.zeros_like(P)
        logP[maskP] = np.log(P[maskP])

        # Initialize input distribution uniformly
        p = np.ones(n, dtype=float) / float(n)

        # Constants and numerical guards
        ln2 = math.log(2.0)
        tiny = 1e-300  # to avoid log(0)
        tol = 1e-10
        max_iter = 10000

        I_old = 0.0

        # Blahut–Arimoto iterative update
        for _ in range(max_iter):
            q = P @ p
            q = np.maximum(q, tiny)
            logq = np.log(q)

            # s_j = sum_i P_ij * (logP_ij - logq_i)
            s = np.sum(np.where(maskP, P * (logP - logq[:, None]), 0.0), axis=0)

            # multiplicative update: t_j = log p_j + s_j  => p_new propto exp(t_j)
            t = np.log(np.maximum(p, tiny)) + s
            tmax = np.max(t)
            ex = np.exp(t - tmax)
            denom = np.sum(ex)
            if denom <= 0 or not np.isfinite(denom):
                return None
            p_new = ex / denom

            # compute mutual information for new distribution (nats)
            q_new = P @ p_new
            q_new = np.maximum(q_new, tiny)
            logq_new = np.log(q_new)
            s_new = np.sum(np.where(maskP, P * (logP - logq_new[:, None]), 0.0), axis=0)
            I_new = float(np.dot(p_new, s_new))

            if abs(I_new - I_old) <= tol:
                p = p_new
                I_old = I_new
                break

            p = p_new
            I_old = I_new
        else:
            # reached max iterations; continue with latest p
            pass

        # Final capacity computation (using entropies) in nats
        y = P @ p
        H_Y = -float(np.sum(np.where(y > 0.0, y * np.log(y), 0.0)))
        H_Y_given_X = -float(np.dot(p, np.sum(np.where(maskP, P * logP, 0.0), axis=0)))
        I_nats = H_Y - H_Y_given_X
        if not np.isfinite(I_nats):
            return None

        C_bits = I_nats / ln2
        # Tolerate tiny negative rounding
        if C_bits < 0 and C_bits > -1e-12:
            C_bits = 0.0

        # Clean up p and renormalize to be a valid distribution
        p = np.maximum(p, 0.0)
        s = float(np.sum(p))
        if s <= 0 or not np.isfinite(s):
            return None
        p = (p / s).tolist()

        return {"x": p, "C": float(C_bits)}