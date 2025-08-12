from typing import Any
import numpy as np

class Solver:
    def solve(self, problem: dict, **kwargs) -> dict:
        """
        Solve the MDP using Policy Iteration.

        Policy Iteration alternates between two steps:
        1. Policy Evaluation: Given a policy, calculate its value function by
           solving a system of linear equations.
        2. Policy Improvement: Greedily update the policy based on the new
           value function.

        This method typically converges in far fewer iterations than Value
        Iteration, making it faster for problems where VI converges slowly.
        """
        num_states = problem["num_states"]
        gamma = problem["discount"]
        transitions = np.array(problem["transitions"])  # Shape (S, A, S')
        rewards = np.array(problem["rewards"])          # Shape (S, A, S')

        # 1. Pre-compute expected rewards R(s, a) for efficiency.
        R_sa = np.sum(transitions * rewards, axis=2)  # Shape (S, A)

        # 2. Initialize with an arbitrary policy (e.g., action 0 for all states).
        policy = np.zeros(num_states, dtype=int)
        
        # A safe upper bound on iterations; PI usually converges much faster.
        max_iter = 200 
        for _ in range(max_iter):
            # 3. Policy Evaluation: Solve the linear system (I - γP^π)V = R^π
            
            # Get the transition matrix P^π and reward vector R^π for the current policy.
            # Using advanced indexing for a fully vectorized operation.
            R_pi = R_sa[np.arange(num_states), policy]
            P_pi = transitions[np.arange(num_states), policy, :]
            
            # Construct the system matrix for the Bellman equations.
            system_matrix = np.eye(num_states) - gamma * P_pi
            
            # Solve for V^π. This is the most expensive step.
            V = np.linalg.solve(system_matrix, R_pi)

            # 4. Policy Improvement: Find a better policy using the new V.
            # Q(s, a) = R(s, a) + γ * Σ_s' P(s'|s,a) * V(s')
            Q_sa = R_sa + gamma * (transitions @ V)
            
            new_policy = np.argmax(Q_sa, axis=1)

            # 5. Check for convergence: If the policy is stable, we are done.
            if np.array_equal(new_policy, policy):
                break
            
            policy = new_policy

        return {
            "value_function": V.tolist(),
            "policy": policy.tolist()
        }