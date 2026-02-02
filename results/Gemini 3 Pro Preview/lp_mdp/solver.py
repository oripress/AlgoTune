import numpy as np
from numba import njit
from typing import Any

@njit(fastmath=True, cache=True)
def policy_iteration(num_states, num_actions, gamma, transitions, rewards):
    # Precompute expected immediate rewards: R[s, a] = sum_{s'} P(s'|s,a) * R(s,a,s')
    R = np.zeros((num_states, num_actions), dtype=np.float64)
    for s in range(num_states):
        for a in range(num_actions):
            val = 0.0
            for sp in range(num_states):
                val += transitions[s, a, sp] * rewards[s, a, sp]
            R[s, a] = val
            
    policy = np.zeros(num_states, dtype=np.int32)
    V = np.zeros(num_states, dtype=np.float64)
    I = np.eye(num_states, dtype=np.float64)
    
    # Pre-allocate A_mat and b_vec
    A_mat = np.zeros((num_states, num_states), dtype=np.float64)
    b_vec = np.zeros(num_states, dtype=np.float64)
    
    for _ in range(100):
        # 1. Policy Evaluation
        # Reset A_mat to I
        # A_mat[:] = I # This copies I into A_mat
        # Or manually:
        for i in range(num_states):
            for j in range(num_states):
                A_mat[i, j] = I[i, j]
        
        # Construct A and b
        for s in range(num_states):
            a = policy[s]
            b_vec[s] = R[s, a]
            for sp in range(num_states):
                A_mat[s, sp] -= gamma * transitions[s, a, sp]
        
        # Solve linear system
        V = np.linalg.solve(A_mat, b_vec)
        
        # 2. Policy Improvement
        new_policy = np.zeros(num_states, dtype=np.int32)
        policy_stable = True
        
        for s in range(num_states):
            # We iterate 0..nA-1 to match reference tie-breaking
            best_a = 0
            # Calculate Q(s, 0)
            val_0 = R[s, 0]
            for sp in range(num_states):
                val_0 += gamma * transitions[s, 0, sp] * V[sp]
            best_val = val_0
            
            for a in range(1, num_actions):
                val = R[s, a]
                for sp in range(num_states):
                    val += gamma * transitions[s, a, sp] * V[sp]
                
                if val > best_val + 1e-8:
                    best_val = val
                    best_a = a
            
            new_policy[s] = best_a
            if best_a != policy[s]:
                policy_stable = False
        
        if policy_stable:
            break
        policy = new_policy

    return V, policy

class Solver:
    def __init__(self):
        # Trigger compilation with dummy data
        dummy_S = 1
        dummy_A = 1
        dummy_gamma = 0.9
        dummy_T = np.ones((1, 1, 1), dtype=np.float64)
        dummy_R = np.zeros((1, 1, 1), dtype=np.float64)
        policy_iteration(dummy_S, dummy_A, dummy_gamma, dummy_T, dummy_R)

    def solve(self, problem: dict[str, Any]) -> dict[str, list[float]]:
        num_states = problem["num_states"]
        num_actions = problem["num_actions"]
        gamma = problem["discount"]
        
        transitions = np.array(problem["transitions"], dtype=np.float64)
        rewards = np.array(problem["rewards"], dtype=np.float64)
        
        V, policy = policy_iteration(num_states, num_actions, gamma, transitions, rewards)
        
        return {"value_function": V.tolist(), "policy": policy.tolist()}