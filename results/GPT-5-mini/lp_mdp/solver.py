import numpy as np
from typing import Any, Dict, List

class Solver:
    def solve(self, problem: Dict[str, Any], **kwargs) -> Dict[str, List[float]]:
        """
        Solve a discounted MDP using policy iteration with exact policy evaluation.

        Returns:
            {
                "value_function": [V(s) ...]  # list of floats, length num_states
                "policy": [a(s) ...]          # list of ints, length num_states
            }

        The final policy is chosen by, for each state s, selecting the action a that
        maximizes RHS = sum_{s'} P(s'|s,a) * (r(s,a,s') + gamma * V[s']) using the same
        tie-breaking (first action within 1e-8) as the reference implementation.
        """
        # Parse inputs
        S = int(problem["num_states"])
        A = int(problem["num_actions"])
        gamma = float(problem["discount"])

        transitions = np.array(problem["transitions"], dtype=float)  # shape (S, A, S)
        rewards = np.array(problem["rewards"], dtype=float)  # shape (S, A, S)

        # Ensure shapes are (S, A, S)
        if transitions.shape != (S, A, S):
            transitions = transitions.reshape((S, A, S))
        if rewards.shape != (S, A, S):
            rewards = rewards.reshape((S, A, S))

        # Initialize policy greedily w.r.t. immediate expected reward (break ties by first)
        expected_immediate = np.sum(transitions * rewards, axis=2)  # shape (S, A)
        policy = np.argmax(expected_immediate, axis=1).astype(int)

        # Policy iteration
        rows = np.arange(S)
        max_iters = max(200, S * 5)
        for _ in range(max_iters):
            # Policy evaluation: solve (I - gamma * P_pi) V = r_pi
            P_pi = transitions[rows, policy, :]  # shape (S, S)
            r_pi = np.sum(transitions[rows, policy, :] * rewards[rows, policy, :], axis=1)

            I_minus = np.eye(S) - gamma * P_pi
            try:
                V = np.linalg.solve(I_minus, r_pi)
            except np.linalg.LinAlgError:
                # fallback to least-squares if singular for numerical reasons
                V, *_ = np.linalg.lstsq(I_minus, r_pi, rcond=None)

            # Policy improvement: greedy wrt Q(s,a) = sum_{s'} P * (r + gamma V)
            rhs = rewards + gamma * V[np.newaxis, np.newaxis, :]
            Q = np.sum(transitions * rhs, axis=2)  # shape (S, A)

            # Build new policy using the same tie-breaking as reference
            new_policy = np.zeros(S, dtype=int)
            for s in range(S):
                best_a = 0
                best_rhs = -1e10  # same initial value as reference
                for a in range(A):
                    val = Q[s, a]
                    if val > best_rhs + 1e-8:
                        best_rhs = val
                        best_a = a
                new_policy[s] = best_a

            if np.array_equal(new_policy, policy):
                break
            policy = new_policy

        # Final evaluation to ensure V corresponds to final policy
        P_pi = transitions[rows, policy, :]
        r_pi = np.sum(transitions[rows, policy, :] * rewards[rows, policy, :], axis=1)
        I_minus = np.eye(S) - gamma * P_pi
        try:
            V = np.linalg.solve(I_minus, r_pi)
        except np.linalg.LinAlgError:
            V, *_ = np.linalg.lstsq(I_minus, r_pi, rcond=None)

        # Recover policy from final V using reference's method (to match tie-breaking exactly)
        rhs = rewards + gamma * V[np.newaxis, np.newaxis, :]
        Q = np.sum(transitions * rhs, axis=2)
        final_policy: List[int] = []
        for s in range(S):
            best_a = 0
            best_rhs = -1e10
            for a in range(A):
                val = Q[s, a]
                if val > best_rhs + 1e-8:
                    best_rhs = val
                    best_a = a
            final_policy.append(int(best_a))

        # Ensure Python native floats/ints
        value_function = [float(x) for x in V.tolist()]

        return {"value_function": value_function, "policy": final_policy}