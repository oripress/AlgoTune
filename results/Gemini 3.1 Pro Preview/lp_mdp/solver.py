import numpy as np
from numba import njit
from typing import Any

@njit(cache=True)
def solve_mdp(num_states, num_actions, gamma, transitions, rewards):
    expected_reward = np.zeros((num_states, num_actions))
    for s in range(num_states):
        for a in range(num_actions):
            for sp in range(num_states):
                expected_reward[s, a] += transitions[s, a, sp] * rewards[s, a, sp]
                
    policy = np.zeros(num_states, dtype=np.int64)
    I = np.eye(num_states)
    
    while True:
        P_pi = np.zeros((num_states, num_states))
        R_pi = np.zeros(num_states)
        for s in range(num_states):
            a = policy[s]
            P_pi[s, :] = transitions[s, a, :]
            R_pi[s] = expected_reward[s, a]
            
        V = np.linalg.solve(I - gamma * P_pi, R_pi)
        
        Q = np.zeros((num_states, num_actions))
        for s in range(num_states):
            for a in range(num_actions):
                Q[s, a] = expected_reward[s, a]
                for sp in range(num_states):
                    Q[s, a] += gamma * transitions[s, a, sp] * V[sp]
                    
        new_policy = np.zeros(num_states, dtype=np.int64)
        for s in range(num_states):
            best_a = 0
            best_q = Q[s, 0]
            for a in range(1, num_actions):
                if Q[s, a] > best_q:
                    best_q = Q[s, a]
                    best_a = a
            new_policy[s] = best_a
            
        same = True
        for s in range(num_states):
            if new_policy[s] != policy[s]:
                same = False
                break
                
        if same:
            break
        policy = new_policy
        
    final_policy = np.zeros(num_states, dtype=np.int64)
    best_rhs = np.full(num_states, -1e10)
    for a in range(num_actions):
        for s in range(num_states):
            rhs_value = Q[s, a]
            if rhs_value > best_rhs[s] + 1e-8:
                final_policy[s] = a
                best_rhs[s] = rhs_value
                
    return V, final_policy

class Solver:
    def __init__(self):
        # Trigger Numba compilation
        dummy_transitions = np.zeros((1, 1, 1))
        dummy_rewards = np.zeros((1, 1, 1))
        solve_mdp(1, 1, 0.9, dummy_transitions, dummy_rewards)

    def solve(self, problem: dict[str, Any], **kwargs) -> dict[str, Any]:
        num_states = problem["num_states"]
        num_actions = problem["num_actions"]
        gamma = problem["discount"]

        transitions = np.array(problem["transitions"])
        rewards = np.array(problem["rewards"])

        V, final_policy = solve_mdp(num_states, num_actions, gamma, transitions, rewards)
        
        return {"value_function": V.tolist(), "policy": final_policy.tolist()}