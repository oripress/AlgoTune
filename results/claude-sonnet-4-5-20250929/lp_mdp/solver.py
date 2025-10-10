import numpy as np
import highspy
from typing import Any

class Solver:
    def solve(self, problem: dict[str, Any]) -> dict[str, list]:
        """
        Solve the MDP using HiGHS LP solver directly.
        
        LP formulation:
        min sum(V_s)
        s.t. V_s - gamma * sum_{s'} P(s'|s,a) * V_{s'} >= sum_{s'} P(s'|s,a) * r(s,a,s')
        """
        num_states = problem["num_states"]
        num_actions = problem["num_actions"]
        gamma = problem["discount"]
        
        transitions = np.array(problem["transitions"], dtype=np.float64)  # (S, A, S)
        rewards = np.array(problem["rewards"], dtype=np.float64)  # (S, A, S)
        
        # Construct LP
        # Construct LP
        num_constraints = num_states * num_actions

        # Objective: minimize sum of V
        c = np.ones(num_states, dtype=np.float64)

        # Build constraint matrix fully vectorized
        # Flatten to (S*A, S) format
        transitions_flat = transitions.reshape(num_constraints, num_states)
        rewards_flat = rewards.reshape(num_constraints, num_states)
        
        # A[i,s] = 1 if constraint i corresponds to state s, else -gamma * P(s'|s,a)
        A = -gamma * transitions_flat.copy()
        state_indices = np.repeat(np.arange(num_states), num_actions)
        A[np.arange(num_constraints), state_indices] += 1.0
        
        # b[i] = sum over s' of P(s'|s,a) * r(s,a,s')
        b = np.sum(transitions_flat * rewards_flat, axis=1)
        
        # Create HiGHS model
        h = highspy.Highs()
        h.setOptionValue("output_flag", False)
        h.setOptionValue("presolve", "on")
        
        # Add variables (V values, unbounded)
        inf = highspy.kHighsInf
        for i in range(num_states):
            h.addVar(-inf, inf)
        
        # Set objective coefficients
        h.changeColsCost(num_states, np.arange(num_states, dtype=np.int32), c)
        
        # Add constraints (>= type, so lower bound = b[i], upper bound = inf)
        for i in range(num_constraints):
            # Get non-zero coefficients for this constraint
            row_indices = np.where(np.abs(A[i, :]) > 1e-12)[0]
            row_values = A[i, row_indices]
            
            h.addRow(b[i], inf, len(row_indices), 
                    row_indices.astype(np.int32), row_values)
        
        # Solve
        h.run()
        
        # Get solution
        solution = h.getSolution()
        val_func = np.array(solution.col_value[:num_states])
        
        # Extract policy
        Q_values = np.sum(transitions * (rewards + gamma * val_func[np.newaxis, np.newaxis, :]), axis=2)
        policy = np.argmax(Q_values, axis=1).tolist()
        
        return {"value_function": val_func.tolist(), "policy": policy}