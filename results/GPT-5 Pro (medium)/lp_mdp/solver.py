from typing import Any, Dict, List

import numpy as np
import cvxpy as cp


class Solver:
    def solve(self, problem: Dict[str, Any], **kwargs: Any) -> Dict[str, List[float]]:
        """
        Solve discounted MDP via LP:
          minimize sum_s V_s
          subject to V_s >= sum_{s'} P(s'|s,a) * (r(s,a,s') + gamma * V_{s'}) for all (s,a)

        We build a single linear inequality A @ V >= b where each row corresponds to (s,a):
          A_row = e_s^T - gamma * P_{s,a}^T
          b_row = sum_{s'} P_{s,a}(s') * r(s,a,s')

        Policy is recovered by selecting, for each state, the action that maximizes the RHS,
        with the same strict tolerance rule as the reference: update only if > best + 1e-8.
        """
        S = int(problem["num_states"])
        A = int(problem["num_actions"])
        gamma = float(problem["discount"])

        # Convert to numpy arrays
        P = np.asarray(problem["transitions"], dtype=np.float64)  # (S, A, S)
        R = np.asarray(problem["rewards"], dtype=np.float64)      # (S, A, S)

        if P.shape != (S, A, S) or R.shape != (S, A, S):
            # Basic shape guard; return trivial solution if malformed
            return {"value_function": [0.0] * S, "policy": [0] * S}

        # Expected immediate reward per (s,a): sum_{s'} P(s'|s,a) * R(s,a,s')
        # Shape: (S, A)
        R_sa = np.einsum("sas,sas->sa", P, R)

        # Build A matrix and b vector for A @ V >= b
        # Flatten (s,a) pairs into rows
        P_flat = P.reshape(S * A, S)  # rows correspond to (s,a)
        A_mat = -gamma * P_flat.copy()
        # Add +1 at column corresponding to state s for each (s,a) row
        s_index = np.repeat(np.arange(S, dtype=np.int64), A)
        A_mat[np.arange(S * A), s_index] += 1.0

        b_vec = R_sa.reshape(S * A)

        # Define and solve LP
        V = cp.Variable(S, name="V")
        constraints = [A_mat @ V >= b_vec]
        objective = cp.Minimize(cp.sum(V))
        prob = cp.Problem(objective, constraints)

        try:
            prob.solve(solver=cp.SCS, eps=1e-5, verbose=False)
        except Exception:
            return {"value_function": [0.0] * S, "policy": [0] * S}

        if V.value is None:
            return {"value_function": [0.0] * S, "policy": [0] * S}

        val = np.asarray(V.value, dtype=np.float64).reshape(S)

        # Recover policy: for each s, pick a maximizing action with strict tolerance
        # Compute RHS Q(s,a) = sum_{s'} P * (R + gamma * V)
        rhs_all = np.sum(P * (R + gamma * val[None, None, :]), axis=2)  # shape (S, A)

        policy: List[int] = []
        for s in range(S):
            best_a = 0
            best_rhs = -1e10
            for a in range(A):
                rhs = float(rhs_all[s, a])
                if rhs > best_rhs + 1e-8:
                    best_rhs = rhs
                    best_a = a
            policy.append(best_a)

        return {"value_function": val.tolist(), "policy": policy}