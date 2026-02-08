import numpy as np
from numba import njit

@njit(cache=True)
def _policy_iteration(P, r_sa, gamma, num_states, num_actions):
    """Full policy iteration in numba."""
    policy = np.empty(num_states, dtype=np.int64)
    
    # Initialize policy greedily from r_sa
    for s in range(num_states):
        best_a = 0
        best_r = r_sa[s, 0]
        for a in range(1, num_actions):
            if r_sa[s, a] > best_r:
                best_r = r_sa[s, a]
                best_a = a
        policy[s] = best_a
    
    A_mat = np.empty((num_states, num_states))
    r_pi = np.empty(num_states)
    P_flat = P.reshape(num_states * num_actions, num_states)
    
    V = np.zeros(num_states)
    
    for iteration in range(1000):
        # Build system for current policy
        for s in range(num_states):
            a = policy[s]
            r_pi[s] = r_sa[s, a]
            for sp in range(num_states):
                A_mat[s, sp] = -gamma * P[s, a, sp]
            A_mat[s, s] += 1.0
        
        # Solve (I - gamma * P_pi) V = r_pi
        V = np.linalg.solve(A_mat, r_pi)
        
        # Policy improvement using matrix multiply
        PV = P_flat @ V  # (S*A,)
        
        changed = False
        for s in range(num_states):
            best_a = 0
            best_q = r_sa[s, 0] + gamma * PV[s * num_actions]
            for a in range(1, num_actions):
                q = r_sa[s, a] + gamma * PV[s * num_actions + a]
                if q > best_q:
                    best_q = q
                    best_a = a
            if best_a != policy[s]:
                policy[s] = best_a
                changed = True
        
        if not changed:
            break
    
    # Extract policy with reference tie-breaking (1e-8 tolerance)
    PV = P_flat @ V
    final_policy = np.empty(num_states, dtype=np.int64)
    for s in range(num_states):
        best_a = 0
        best_q = -1e300
        for a in range(num_actions):
            q = r_sa[s, a] + gamma * PV[s * num_actions + a]
            if q > best_q + 1e-8:
                best_q = q
                best_a = a
        final_policy[s] = best_a
    
    return V, final_policy

@njit(cache=True)
def _value_iteration(P_flat, r_sa, gamma, num_states, num_actions):
    V = np.zeros(num_states)
    for _ in range(100000):
        PV = P_flat @ V
        V_new = np.empty(num_states)
        max_diff = 0.0
        for s in range(num_states):
            best = -1e300
            for a in range(num_actions):
                q = r_sa[s, a] + gamma * PV[s * num_actions + a]
                if q > best:
                    best = q
            V_new[s] = best
            d = abs(best - V[s])
            if d > max_diff:
                max_diff = d
        if max_diff < 1e-12:
            V = V_new
            break
        V = V_new
    
    PV = P_flat @ V
    policy = np.empty(num_states, dtype=np.int64)
    for s in range(num_states):
        best_a = 0
        best_q = -1e300
        for a in range(num_actions):
            q = r_sa[s, a] + gamma * PV[s * num_actions + a]
            if q > best_q + 1e-8:
                best_q = q
                best_a = a
        policy[s] = best_a
    return V, policy

@njit(cache=True)
def _compute_r_sa(P, R_full, num_states, num_actions):
    r_sa = np.empty((num_states, num_actions))
    for s in range(num_states):
        for a in range(num_actions):
            val = 0.0
            for sp in range(num_states):
                val += P[s, a, sp] * R_full[s, a, sp]
            r_sa[s, a] = val
    return r_sa

# Trigger compilation at import time
def _warmup():
    P = np.array([[[0.5, 0.5], [0.5, 0.5]], [[0.5, 0.5], [0.5, 0.5]]])
    R = np.array([[[1.0, 0.0], [0.0, 1.0]], [[1.0, 0.0], [0.0, 1.0]]])
    r_sa = _compute_r_sa(P, R, 2, 2)
    _policy_iteration(P, r_sa, 0.9, 2, 2)
    P_flat = P.reshape(4, 2)
    _value_iteration(P_flat, r_sa, 0.9, 2, 2)

_warmup()

class Solver:
    def solve(self, problem, **kwargs):
        num_states = problem["num_states"]
        num_actions = problem["num_actions"]
        gamma = problem["discount"]
        
        P = np.asarray(problem["transitions"], dtype=np.float64)
        R_full = np.asarray(problem["rewards"], dtype=np.float64)
        
        r_sa = _compute_r_sa(P, R_full, num_states, num_actions)
        
        if num_states <= 800:
            V, policy_out = _policy_iteration(P, r_sa, gamma, num_states, num_actions)
        else:
            P_flat = P.reshape(num_states * num_actions, num_states)
            V, policy_out = _value_iteration(P_flat, r_sa, gamma, num_states, num_actions)
        
        return {
            "value_function": V.tolist(),
            "policy": policy_out.tolist()
        }