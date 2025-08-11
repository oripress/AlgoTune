from typing import Any
import numpy as np
from scipy.optimize import linprog

class Solver:
    def solve(self, problem: dict[str, Any]) -> list[int]:
        """
        Solve the DAP problem using linear programming.
        """
        T = problem["T"]
        N = problem["N"]
        prices = problem["prices"]
        capacities = problem["capacities"]
        probs = problem["probs"]
        
        # Create decision variables: x[t*N + i] represents offering product i at time t
        num_vars = T * N
        
        # Objective function coefficients (negative because linprog minimizes)
        c = []
        for t in range(T):
            for i in range(N):
                c.append(-prices[i] * probs[t][i])
        c = np.array(c)
        
        # Inequality constraints: A_ub @ x <= b_ub
        A_ub = []
        b_ub = []
        
        # Constraint 1: At most one product per period
        for t in range(T):
            constraint = np.zeros(num_vars)
            for i in range(N):
                constraint[t * N + i] = 1
            A_ub.append(constraint)
            b_ub.append(1)
        
        # Constraint 2: Capacity limits
        for i in range(N):
            constraint = np.zeros(num_vars)
            for t in range(T):
                constraint[t * N + i] = 1
            A_ub.append(constraint)
            b_ub.append(capacities[i])
        
        A_ub = np.array(A_ub)
        b_ub = np.array(b_ub)
        
        # Variable bounds: 0 <= x <= 1
        bounds = [(0, 1) for _ in range(num_vars)]
        
        # Solve
        result = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
        
        if not result.success:
            return [-1] * T
        
        # Extract solution (round to nearest integer)
        x = result.x
        offer = []
        for t in range(T):
            chosen = -1
            max_val = 0.5  # Threshold for rounding
            for i in range(N):
                if x[t * N + i] > max_val:
                    max_val = x[t * N + i]
                    chosen = i
            offer.append(chosen)
        
        return offer