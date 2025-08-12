import numpy as np
from typing import Any
from numba import jit, prange
import scipy.linalg

# JIT-compiled policy iteration function with aggressive optimizations
@jit(nopython=True, fastmath=True, cache=True, parallel=False)
def policy_iteration_numba_optimized(transitions, expected_rewards, gamma, num_states, num_actions):
    # Initialize policy deterministically (all zeros)
    policy = np.zeros(num_states, dtype=np.int32)
    
    # Initialize value function
    V = np.zeros(num_states)
    
    # Pre-allocate matrices for better memory access
    P_pi = np.zeros((num_states, num_states))
    R_pi = np.zeros(num_states)
    A = np.eye(num_states)
    
    # Reduce max iterations - policy iteration typically converges in < 20 iterations
    max_iterations = 50
    
    for iteration in range(max_iterations):
        # Policy evaluation: build P^π and R^π
        # Clear matrices
        P_pi.fill(0.0)
        R_pi.fill(0.0)
        
        for s in range(num_states):
            a = policy[s]
            for sp in range(num_states):
                P_pi[s, sp] = transitions[s, a, sp]
            R_pi[s] = expected_rewards[s, a]
        
        # Build A = (I - γ * P^π) in-place
        for s in range(num_states):
            for sp in range(num_states):
                A[s, sp] = -gamma * P_pi[s, sp]
            A[s, s] += 1.0
        
        # Use LU factorization for better numerical stability
        V_new = np.linalg.solve(A, R_pi)
        
        # Policy improvement with early termination
        policy_changed = False
        
        # Pre-compute gamma * V_new for reuse
        gamma_V = gamma * V_new
        
        for s in range(num_states):
            best_value = -np.inf
            best_action = 0
            
            for a in range(num_actions):
                # Q(s,a) = R(s,a) + γ * sum_{s'} P(s'|s,a) * V_{s'}
                q_value = expected_rewards[s, a]
                for sp in range(num_states):
                    q_value += transitions[s, a, sp] * gamma_V[sp]
                
                if q_value > best_value + 1e-12:  # tighter tolerance
                    best_value = q_value
                    best_action = a
            
            if policy[s] != best_action:
                policy_changed = True
                policy[s] = best_action
        
        if not policy_changed:
            break
        
        V = V_new
    
    # Final policy evaluation with optimized memory access
    P_pi.fill(0.0)
    R_pi.fill(0.0)
    
    for s in range(num_states):
        a = policy[s]
        for sp in range(num_states):
            P_pi[s, sp] = transitions[s, a, sp]
        R_pi[s] = expected_rewards[s, a]
    
    # Build final A matrix in-place
    for s in range(num_states):
        for sp in range(num_states):
            A[s, sp] = -gamma * P_pi[s, sp]
        A[s, s] += 1.0
    
    V_final = np.linalg.solve(A, R_pi)
    
    return V_final, policy

class Solver:
    def solve(self, problem: dict[str, Any]) -> dict[str, Any]:
        """
        Solve the MDP using ultra-optimized policy iteration with Numba JIT compilation
        and aggressive numerical optimizations.
        """
        num_states = problem["num_states"]
        num_actions = problem["num_actions"]
        gamma = problem["discount"]
        
        # Convert to numpy arrays with optimal memory layout
        transitions = np.array(problem["transitions"], dtype=np.float64, order='C')
        rewards = np.array(problem["rewards"], dtype=np.float64, order='C')
        
        # Pre-compute expected rewards with optimized memory access
        expected_rewards = np.sum(transitions * rewards, axis=2)  # shape: (S, A)
        
        # Use Numba-compiled policy iteration
        V_final, policy = policy_iteration_numba_optimized(
            transitions, expected_rewards, gamma, num_states, num_actions
        )
        
        return {"value_function": V_final.tolist(), "policy": policy.tolist()}