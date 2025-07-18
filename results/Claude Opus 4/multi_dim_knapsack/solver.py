from typing import Any, List, Tuple, NamedTuple
import cvxpy as cp
import numpy as np

class MultiDimKnapsackInstance(NamedTuple):
    value: List[int]
    demand: List[List[int]]
    supply: List[int]

class Solver:
    def solve(self, problem, **kwargs) -> Any:
        """Solve the multi-dimensional knapsack problem using CVXPY."""
        if not isinstance(problem, MultiDimKnapsackInstance):
            try:
                problem = MultiDimKnapsackInstance(*problem)
            except Exception:
                return []
        
        n = len(problem.value)
        k = len(problem.supply)
        
        # Convert to numpy arrays
        values = np.array(problem.value)
        demands = np.array(problem.demand)
        supplies = np.array(problem.supply)
        
        # Decision variables
        x = cp.Variable(n, boolean=True)
        
        # Objective
        objective = cp.Maximize(values @ x)
        
        # Constraints
        constraints = [demands.T @ x <= supplies]
        
        # Solve
        prob = cp.Problem(objective, constraints)
        prob.solve(solver=cp.ECOS_BB, verbose=False)
        
        if prob.status == cp.OPTIMAL:
            return [i for i in range(n) if x.value[i] > 0.5]
        return []