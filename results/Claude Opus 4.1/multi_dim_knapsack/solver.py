from typing import NamedTuple, Any
import numpy as np
from ortools.sat.python import cp_model

class MultiDimKnapsackInstance(NamedTuple):
    value: list
    demand: list
    supply: list

MultiKnapsackSolution = list

class Solver:
    def __init__(self):
        # Pre-initialize OR-Tools
        self.model_cache = {}
        
    def solve(self, problem: Any, **kwargs) -> Any:
        """Optimized Multi-Dimensional Knapsack solver."""
        if not isinstance(problem, MultiDimKnapsackInstance):
            try:
                problem = MultiDimKnapsackInstance(*problem)
            except Exception:
                return []

        n = len(problem.value)
        k = len(problem.supply)
        
        if n == 0 or k == 0:
            return []
        
        # For very small problems, use brute force with pruning
        if n <= 12:
            best_value = 0
            best_solution = []
            
            # Use bit manipulation for subset generation
            for mask in range(1 << n):
                selected = []
                total_value = 0
                feasible = True
                
                # Check this subset
                usage = [0] * k
                for i in range(n):
                    if mask & (1 << i):
                        selected.append(i)
                        total_value += problem.value[i]
                        for j in range(k):
                            usage[j] += problem.demand[i][j]
                            if usage[j] > problem.supply[j]:
                                feasible = False
                                break
                    if not feasible:
                        break
                
                if feasible and total_value > best_value:
                    best_value = total_value
                    best_solution = selected
            
            return best_solution
        
        # For larger problems, use OR-Tools with aggressive optimization
        model = cp_model.CpModel()
        x = [model.NewBoolVar(f"x_{i}") for i in range(n)]
        
        # Add constraints with early filtering
        for r in range(k):
            # Filter out items that can never fit
            valid_items = [i for i in range(n) if problem.demand[i][r] <= problem.supply[r]]
            if valid_items:
                model.Add(sum(x[i] * problem.demand[i][r] for i in valid_items) <= problem.supply[r])
        
        # Set objective
        model.Maximize(sum(x[i] * problem.value[i] for i in range(n)))
        
        # Add greedy hints
        values = np.array(problem.value, dtype=np.float64)
        demands = np.array(problem.demand, dtype=np.float64)
        supplies = np.array(problem.supply, dtype=np.float64)
        
        # Calculate value density
        densities = np.zeros(n)
        for i in range(n):
            total_demand = 0.0
            for j in range(k):
                if supplies[j] > 0:
                    total_demand += demands[i][j] / supplies[j]
            if total_demand > 0:
                densities[i] = values[i] / total_demand
        
        # Sort by density and add hints
        indices = np.argsort(-densities)
        remaining = supplies.copy()
        hint_solution = []
        
        for idx in indices[:min(n, 20)]:  # Only hint top items
            fits = True
            for j in range(k):
                if demands[idx][j] > remaining[j]:
                    fits = False
                    break
            if fits:
                hint_solution.append(idx)
                for j in range(k):
                    remaining[j] -= demands[idx][j]
                model.AddHint(x[idx], 1)
        
        # Configure solver for maximum speed
        solver = cp_model.CpSolver()
        
        if n <= 50:
            solver.parameters.max_time_in_seconds = 0.5
            solver.parameters.num_search_workers = 1
        elif n <= 100:
            solver.parameters.max_time_in_seconds = 1.0
            solver.parameters.num_search_workers = 2
        else:
            solver.parameters.max_time_in_seconds = 2.0
            solver.parameters.num_search_workers = 4
            
        solver.parameters.linearization_level = 2
        solver.parameters.cp_model_presolve = True
        solver.parameters.cp_model_probing_level = 0  # Disable probing for speed
        solver.parameters.search_branching = cp_model.FIXED_SEARCH
        solver.parameters.use_pb_resolution = False
        solver.parameters.minimize_reduction_during_pb_resolution = False
        
        # Solve
        status = solver.Solve(model)
        
        if status in (cp_model.OPTIMAL, cp_model.FEASIBLE):
            return [i for i in range(n) if solver.Value(x[i])]
        return []