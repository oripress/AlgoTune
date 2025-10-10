from typing import List
from ortools.sat.python import cp_model
import numpy as np
from numba import njit

@njit
def check_feasibility(subset_mask, demand, supply, n):
    """Check if a subset is feasible."""
    k = len(supply)
    for r in range(k):
        total = 0
        for i in range(n):
            if subset_mask[i]:
                total += demand[i, r]
        if total > supply[r]:
            return False
    return True

@njit
def enumerate_solutions(demand, value, supply, n):
    """Enumerate all solutions and find the best."""
    best_value = 0
    best_mask = np.zeros(n, dtype=np.bool_)
    
    # Iterate through all subsets
    for mask_int in range(1 << n):
        subset_mask = np.zeros(n, dtype=np.bool_)
        for i in range(n):
            if mask_int & (1 << i):
                subset_mask[i] = True
        
        if check_feasibility(subset_mask, demand, supply, n):
            total_value = 0
            for i in range(n):
                if subset_mask[i]:
                    total_value += value[i]
            
            if total_value > best_value:
                best_value = total_value
                best_mask = subset_mask.copy()
    
    return best_mask

class Solver:
    def solve(self, problem, **kwargs) -> List[int]:
        """
        Solves the Multi-Dimensional Knapsack Problem with optimizations.
        """
        # Parse input
        if isinstance(problem, tuple) or isinstance(problem, list):
            value, demand, supply = problem
        else:
            return []
        
        n = len(value)
        k = len(supply)
        
        if n == 0:
            return []
        
        # Fast numpy-based preprocessing
        demand_np = np.array(demand, dtype=np.int32)
        value_np = np.array(value, dtype=np.int32)
        supply_np = np.array(supply, dtype=np.int32)
        
        if n <= 23:  # Brute force for very small instances
            best_mask = enumerate_solutions(demand_np, value_np, supply_np, n)
            return [int(i) for i in range(n) if best_mask[i]]
        feasible_mask = np.all(demand_np <= supply_np, axis=1)
        feasible = np.where(feasible_mask)[0]
        
        if len(feasible) == 0:
            return []
        
        # For small problems, use exact enumeration with numba
        if len(feasible) <= 20:
            feasible_demand = demand_np[feasible]
            feasible_value = value_np[feasible]
            best_mask = enumerate_solutions(feasible_demand, feasible_value, supply_np, len(feasible))
            return [int(feasible[i]) for i in range(len(feasible)) if best_mask[i]]
        greedy_sol = self._greedy_heuristic(value_np, demand_np, supply_np, feasible)
        
        # Create model
        model = cp_model.CpModel()
        
        # Decision variables (only for feasible items)
        x = {i: model.NewBoolVar(f"x_{i}") for i in feasible}
        
        # Resource constraints
        for r in range(k):
            model.Add(sum(x[i] * demand_np[i, r] for i in feasible) <= supply_np[r])
        
        # Objective function
        model.Maximize(sum(x[i] * value_np[i] for i in feasible))
        
        # Add greedy solution as hint
        for i in feasible:
            model.AddHint(x[i], 1 if i in greedy_sol else 0)
        
        solver = cp_model.CpSolver()
        solver.parameters.num_search_workers = 4
        solver.parameters.log_search_progress = False
        status = solver.Solve(model)
        
        # Extract solution
        if status in (cp_model.OPTIMAL, cp_model.FEASIBLE):
            return [int(i) for i in feasible if solver.Value(x[i])]
        return [int(i) for i in greedy_sol]
    
    def _greedy_heuristic(self, value_np, demand_np, supply_np, feasible):
        """Fast greedy heuristic."""
        # Calculate efficiency (value per average resource usage)
        avg_demand = demand_np[feasible].mean(axis=1)
        efficiency = value_np[feasible] / (avg_demand + 1e-10)
        
        # Sort by efficiency
        sorted_indices = np.argsort(-efficiency)
        
        solution = []
        remaining = supply_np.copy()
        
        for idx in sorted_indices:
            item = feasible[idx]
            if np.all(demand_np[item] <= remaining):
                solution.append(item)
                remaining -= demand_np[item]
        
        return solution