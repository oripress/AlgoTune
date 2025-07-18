import numpy as np
import cvxpy as cp
from typing import Any

class Solver:
    def __init__(self):
        # Pre-allocate arrays for efficiency
        self.max_states = 1000
        self.max_actions = 100
        
    def solve(self, problem: dict[str, Any]) -> dict[str, list[float]]:
        """
        Solve the MDP using cvxpy with optimized constraint construction.
        """
        num_states = problem["num_states"]
        num_actions = problem["num_actions"]
        gamma = problem["discount"]
        
        # Use float32 for faster computation
        transitions = np.array(problem["transitions"], dtype=np.float32)
        rewards = np.array(problem["rewards"], dtype=np.float32)
        
        # Pre-compute expected rewards
        expected_rewards = np.sum(transitions * rewards, axis=2)
        
        # Define variables
        V = cp.Variable(num_states)
        
        # Vectorized constraint construction
        constraints = []
        
        # Reshape transitions for efficient matrix multiplication
        trans_reshaped = transitions.reshape(num_states * num_actions, num_states)
        exp_rew_flat = expected_rewards.flatten()
        
        # Create all constraints at once
        for i in range(num_states * num_actions):
            s = i // num_actions
            constraints.append(V[s] >= exp_rew_flat[i] + gamma * trans_reshaped[i] @ V)
        
        # Objective
        objective = cp.Minimize(cp.sum(V))
        
        # Solve with optimized parameters
        prob = cp.Problem(objective, constraints)
        prob.solve(solver=cp.ECOS, verbose=False, abstol=1e-6, reltol=1e-6)
        
        if V.value is None:
            return {"value_function": [0.0] * num_states, "policy": [0] * num_states}
        
        # Extract value function
        val_func = V.value.astype(np.float64).tolist()
        
        # Compute policy efficiently using vectorized operations
        V_array = np.array(val_func)
        Q_values = expected_rewards + gamma * np.dot(transitions, V_array)
        policy = np.argmax(Q_values, axis=1).tolist()
        
        return {"value_function": val_func, "policy": policy}