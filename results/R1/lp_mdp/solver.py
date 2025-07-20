import numpy as np

class Solver:
    def solve(self, problem, **kwargs):
        # Handle case where problem is passed as JSON string
        if isinstance(problem, str):
            import json
            problem = json.loads(problem)
        num_states = problem["num_states"]
        num_actions = problem["num_actions"]
        gamma = problem["discount"]
        transitions = np.array(problem["transitions"])
        rewards = np.array(problem["rewards"])
        
        # Precompute expected immediate reward per (s,a)
        b = np.sum(transitions * rewards, axis=2)  # shape (S, A)
        S = num_states
        
        # Initialize policy: all zeros
        policy = np.zeros(S, dtype=int)
        changed = True
        max_iter = 100
        
        for _ in range(max_iter):
            # Build transition matrix and reward vector for current policy
            P_pi = transitions[np.arange(S), policy]  # shape (S, S)
            r_pi = b[np.arange(S), policy]  # shape (S,)
            
            # Solve linear system: (I - γP_pi)V = r_pi
            try:
                # Use direct solver for accuracy
                V = np.linalg.solve(np.eye(S) - gamma * P_pi, r_pi)
            except np.linalg.LinAlgError:
                # Fallback to iterative method if singular matrix
                V = np.zeros(S)
                for _ in range(1000):
                    V_new = r_pi + gamma * P_pi @ V
                    if np.max(np.abs(V_new - V)) < 1e-8:
                        V = V_new
                        break
                    V = V_new
            
            # Compute Q-values for all state-action pairs
            Q = b + gamma * transitions @ V  # shape (S, A)
            
            # Find best actions with tolerance-based tie-breaking
            max_vals = np.max(Q, axis=1, keepdims=True)
            mask = (Q >= max_vals - 1e-8)
            new_policy = np.argmax(mask, axis=1)
            
            # Check for policy convergence
            if np.array_equal(policy, new_policy):
                break
            policy = new_policy
        
        return {"value_function": V.tolist(), "policy": policy.tolist()}