import numpy as np
from scipy.optimize import linprog
from typing import Any

class Solver:
    def solve(self, problem: dict[str, Any]) -> dict[str, Any]:
        """
        Solve the MDP using scipy's linprog for better performance.
        """
        num_states = problem["num_states"]
        num_actions = problem["num_actions"]
        gamma = problem["discount"]
        
        transitions = np.array(problem["transitions"])  # shape: (S, A, S)
        rewards = np.array(problem["rewards"])  # shape: (S, A, S)
        
        # Precompute expected rewards for each (s,a) pair
        # r(s,a) = sum_{s'} p(s'|s,a) * r(s,a,s')
        expected_rewards = np.sum(transitions * rewards, axis=2)  # shape: (S, A)
        
        # Build the LP: minimize sum(V_s)
        # subject to: V_s >= r(s,a) + gamma * sum_{s'} p(s'|s,a) * V_{s'} for all s,a
        
        # Objective: minimize sum of all V_s
        c = np.ones(num_states)
        
        # Build constraint matrix A_ub and b_ub for A_ub @ V <= b_ub
        # We need -V_s + gamma * sum_{s'} p(s'|s,a) * V_{s'} <= -r(s,a)
        num_constraints = num_states * num_actions
        A_ub = np.zeros((num_constraints, num_states))
        b_ub = np.zeros(num_constraints)
        
        constraint_idx = 0
        for s in range(num_states):
            for a in range(num_actions):
                # -V_s + gamma * sum_{s'} p(s'|s,a) * V_{s'}
                A_ub[constraint_idx, s] = -1
                A_ub[constraint_idx, :] += gamma * transitions[s, a, :]
                b_ub[constraint_idx] = -expected_rewards[s, a]
                constraint_idx += 1
        
        # Solve the LP
        result = linprog(c, A_ub=A_ub, b_ub=b_ub, method='highs', options={'disp': False})
        
        if not result.success:
            return {"value_function": [0.0] * num_states, "policy": [0] * num_states}
        
        val_func = result.x.tolist()
        
        # Recover policy by computing Q-values
        Q = expected_rewards + gamma * transitions @ result.x  # shape: (S, A)
        policy = np.argmax(Q, axis=1).tolist()
        
        return {"value_function": val_func, "policy": policy}