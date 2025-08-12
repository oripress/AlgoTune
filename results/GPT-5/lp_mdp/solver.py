from __future__ import annotations

from typing import Any, Dict, List

import numpy as np

class Solver:
    def solve(self, problem: Dict[str, Any], **kwargs) -> Dict[str, List[float]]:
        """
        Solve the discounted MDP via policy iteration (fast and precise), then
        recover a greedy policy using the same tie-breaking as the reference.

        problem keys:
          - num_states: int
          - num_actions: int
          - discount: float
          - transitions: list/array (S, A, S)
          - rewards: list/array (S, A, S)

        Returns:
          - value_function: list of floats length S
          - policy: list of ints length S
        """
        # Extract problem data
        S = int(problem["num_states"])
        A = int(problem["num_actions"])
        gamma = float(problem["discount"])

        # Convert to numpy arrays
        transitions = np.asarray(problem["transitions"], dtype=np.float64)  # (S, A, S)
        rewards = np.asarray(problem["rewards"], dtype=np.float64)  # (S, A, S)

        # Precompute expected immediate rewards r(s,a) = sum_{s'} P * R
        # Shape: (S, A)
        r_sa = np.sum(transitions * rewards, axis=2)

        # Precompute flattened views for fast Q computation
        P_flat = transitions.reshape(S * A, S)
        r_flat = r_sa.reshape(S * A)

        idx = np.arange(S, dtype=np.int64)

        # Helper: evaluate a policy exactly by solving linear system
        # (I - gamma P_pi) V = r_pi
        def evaluate_policy(policy: np.ndarray) -> np.ndarray:
            # P_pi: (S, S), each row s is transitions[s, policy[s], :]
            P_pi = transitions[idx, policy, :]  # advanced indexing yields a copy
            r_pi = r_sa[idx, policy]  # (S,)

            # Build M = I - gamma * P_pi in-place on the local copy
            P_pi *= -gamma
            P_pi.flat[:: S + 1] += 1.0  # add identity to diagonal

            # Solve linear system (I - gamma P_pi) V = r_pi
            V = np.linalg.solve(P_pi, r_pi)
            return V

        # Helper: greedy policy from a value function (reference tie-breaking)
        def greedy_from_value(V: np.ndarray) -> np.ndarray:
            # Compute tmp = P_flat @ V (length S*A)
            tmp = P_flat @ V

            # Scan per-state using flattened buffers to avoid forming Q matrix
            new_pi = np.empty(S, dtype=np.int64)
            thresh = 1e-8
            g = gamma
            base = 0
            for s in range(S):
                best_a = 0
                best_val = r_flat[base] + g * tmp[base]
                base += 1
                for a in range(1, A):
                    qa = r_flat[base] + g * tmp[base]
                    if qa > best_val + thresh:
                        best_val = qa
                        best_a = a
                    base += 1
                new_pi[s] = best_a
            return new_pi

        # Helper: initial greedy policy from immediate rewards r_sa (avoids first matvec)
        def greedy_from_r() -> np.ndarray:
            new_pi = np.empty(S, dtype=np.int64)
            thresh = 1e-8
            for s in range(S):
                row = r_sa[s]
                best_a = 0
                best_val = row[0]
                for a in range(1, A):
                    qa = row[a]
                    if qa > best_val + thresh:
                        best_val = qa
                        best_a = a
                new_pi[s] = best_a
            return new_pi

        # Edge case: if only one action, evaluation suffices
        if A == 1:
            V = evaluate_policy(np.zeros(S, dtype=np.int64))
            final_policy = np.zeros(S, dtype=np.int64)
        else:
            # Initialize policy greedily from immediate rewards (no matvec)
            policy = greedy_from_r()

            # Policy iteration
            max_iters = 1000  # very safe upper bound
            stable = False
            last_new_policy = policy
            for _ in range(max_iters):
                V = evaluate_policy(policy)
                new_policy = greedy_from_value(V)
                last_new_policy = new_policy
                if np.array_equal(new_policy, policy):
                    stable = True
                    break
                policy = new_policy

            # Recover policy using the reference's tie-breaking on the final V
            final_policy = policy if stable else last_new_policy

        # Prepare outputs
        val_func_list = V.tolist()
        policy_list = final_policy.astype(int).tolist()

        # Ensure finite results
        if not np.all(np.isfinite(V)):
            # Fallback to zeros if numerical issue (very unlikely)
            val_func_list = [0.0] * S
            policy_list = [0] * S

        return {"value_function": val_func_list, "policy": policy_list}