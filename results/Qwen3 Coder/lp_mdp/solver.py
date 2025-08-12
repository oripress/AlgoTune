import numpy as np
import cvxpy as cp
from typing import Any, Dict

class Solver:
    def solve(self, problem: Dict[str, Any]) -> Dict[str, Any]:
        """
        Solve the MDP using the standard linear program for discounted MDPs:
        """
        num_states = problem["num_states"]
        num_actions = problem["num_actions"]
        gamma = problem["discount"]
        
        transitions = np.array(problem["transitions"])  # shape: (S, A, S)
        rewards = np.array(problem["rewards"])  # shape: (S, A, S)
        
        # Define variables
        V = cp.Variable(num_states)
        
        # Precompute expected rewards for all state-action pairs
        # Shape: (S, A) where [s,a] entry is sum_{s'} P(s'|s,a) * R(s,a,s')
        expected_rewards = np.sum(transitions * rewards, axis=2)
        
        # Create constraints using matrix operations for better performance
        # Create a single large constraint instead of multiple smaller ones for better performance
        # Reshape transitions and rewards to create all constraints at once
        # Shape: (S*A, S) where each row corresponds to a state-action pair
        T_flat = transitions.reshape(-1, num_states)  # Shape: (S*A, S)
        
        # Compute expected rewards for all state-action pairs at once
        # Shape: (S*A,) where entry [s*A + a] is sum_{s'} P(s'|s,a) * R(s,a,s')
        expected_rewards_flat = np.sum(transitions * rewards, axis=2).reshape(-1)
        
        # Create the constraint matrix for all state-action pairs at once
        # V >= expected_rewards_flat + gamma * (T_flat @ V)
        # This is equivalent to V - gamma * (T_flat @ V) >= expected_rewards_flat
        # Which can be written as (I - gamma * T_flat^T) @ V >= expected_rewards_flat where I is (S*A, S)
        # But we need to be more careful with the dimensions
        # Let's use the original approach but with better vectorization
        constraints = []
        for a in range(num_actions):
            T_a = transitions[:, a, :]
            rhs = expected_rewards[:, a] + gamma * (T_a @ V)
            constraints.append(V >= rhs)
        # Objective: minimize sum of values (primal form)
        objective = cp.Minimize(cp.sum(V))
        
        # Solve the problem with optimized settings
        prob = cp.Problem(objective, constraints)
        
        # Try the fastest solver first
        try:
            prob.solve(solver=cp.CLARABEL, verbose=False)
        except:
            try:
                prob.solve(solver=cp.OSQP, verbose=False)
            except:
                # Fallback to SCS with tighter settings
                prob.solve(solver=cp.SCS, verbose=False, eps=1e-6, max_iters=3000)
        
        if V.value is None:
            return {"value_function": [0.0] * num_states, "policy": [0] * num_states}
            
        val_func = V.value
        
        # Vectorized policy recovery using matrix operations
        # Compute all Q-values at once: Q(s,a) = sum_{s'} P(s'|s,a) * (R(s,a,s') + gamma * V(s'))
        Q_values = np.sum(transitions * (rewards + gamma * val_func[None, None, :]), axis=2)
        
        # Find optimal actions for each state
        policy = np.argmax(Q_values, axis=1).tolist()
        
        return {"value_function": val_func.tolist(), "policy": policy}