import numpy as np
from typing import Any

class Solver:
    def solve(self, problem: dict[str, Any]) -> dict[str, Any]:
        """
        Solve the discounted MDP using fast value iteration.
        Returns the optimal value function and a deterministic optimal policy.
        """
        num_states = problem["num_states"]
        num_actions = problem["num_actions"]
        gamma = problem["discount"]
        transitions = np.array(problem["transitions"], dtype=np.float64)  # shape (S, A, S)
        rewards = np.array(problem["rewards"], dtype=np.float64)        # shape (S, A, S)
        # Choose algorithm based on problem size
        if num_states <= 50:
            # Small MDP: use policy iteration (fast convergence)
            policy = np.zeros(num_states, dtype=np.int64)
            V = np.zeros(num_states, dtype=np.float64)

            max_pi_iter = 100
            for _ in range(max_pi_iter):
                # Policy evaluation
                P_pi = transitions[np.arange(num_states), policy, :]          # (S, S)
                r_pi = np.sum(transitions[np.arange(num_states), policy, :] *
                              rewards[np.arange(num_states), policy, :], axis=1)  # (S,)

                A = np.eye(num_states) - gamma * P_pi
                V = np.linalg.solve(A, r_pi)

                # Policy improvement
                Q = np.sum(transitions * (rewards + gamma * V[np.newaxis, np.newaxis, :]), axis=2)  # (S, A)
                new_policy = np.argmax(Q, axis=1)

                if np.array_equal(new_policy, policy):
                    break
                policy = new_policy
        else:
            # Large MDP: use fast value iteration
            V = np.zeros(num_states, dtype=np.float64)
            max_vi_iter = 10000
            tol = 1e-9
            for _ in range(max_vi_iter):
                # Compute Q-values for all actions
                Q = np.sum(transitions * (rewards + gamma * V[np.newaxis, np.newaxis, :]), axis=2)  # (S, A)
                V_new = np.max(Q, axis=1)
                if np.max(np.abs(V_new - V)) < tol:
                    V = V_new
                    break
                V = V_new

            # Derive policy from final value function
            Q = np.sum(transitions * (rewards + gamma * V[np.newaxis, np.newaxis, :]), axis=2)  # (S, A)
            policy = np.argmax(Q, axis=1)

        # Ensure policy is a plain Python list for output
        policy = policy.tolist()

        return {"value_function": V.tolist(), "policy": policy}