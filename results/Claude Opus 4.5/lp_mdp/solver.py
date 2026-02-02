import numpy as np
from numba import njit

@njit(cache=True, fastmath=True)
def value_iteration_numba(transitions, expected_rewards, gamma, tol=1e-10, max_iter=10000):
    num_states = transitions.shape[0]
    num_actions = transitions.shape[1]
    V = np.zeros(num_states)
    
    for _ in range(max_iter):
        max_diff = 0.0
        for s in range(num_states):
            v_old = V[s]
            max_q = -1e100
            for a in range(num_actions):
                q = expected_rewards[s, a]
                for sp in range(num_states):
                    q += gamma * transitions[s, a, sp] * V[sp]
                if q > max_q:
                    max_q = q
            V[s] = max_q
            diff = abs(max_q - v_old)
            if diff > max_diff:
                max_diff = diff
        
        if max_diff < tol:
            break
    
    return V

@njit(cache=True, fastmath=True)
def extract_policy(transitions, expected_rewards, gamma, V):
    num_states = transitions.shape[0]
    num_actions = transitions.shape[1]
    policy = np.empty(num_states, dtype=np.int64)
    for s in range(num_states):
        best_a = 0
        best_q = -1e100
        for a in range(num_actions):
            q = expected_rewards[s, a]
            for sp in range(num_states):
                q += gamma * transitions[s, a, sp] * V[sp]
            if q > best_q + 1e-8:
                best_q = q
                best_a = a
        policy[s] = best_a
    return policy

class Solver:
    def solve(self, problem, **kwargs):
        num_states = problem["num_states"]
        num_actions = problem["num_actions"]
        gamma = problem["discount"]
        
        transitions = np.ascontiguousarray(problem["transitions"], dtype=np.float64)
        rewards = np.asarray(problem["rewards"], dtype=np.float64)
        
        expected_rewards = np.ascontiguousarray(np.sum(transitions * rewards, axis=2))
        
        V = value_iteration_numba(transitions, expected_rewards, gamma)
        policy = extract_policy(transitions, expected_rewards, gamma, V)
        
        return {"value_function": V.tolist(), "policy": policy.tolist()}