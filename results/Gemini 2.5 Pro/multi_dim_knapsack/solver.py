from typing import Any
from ortools.linear_solver import pywraplp

class Solver:
    def solve(self, problem: tuple, **kwargs) -> Any:
        """
        Switches the MIP backend to BOP (Boolean Optimization Problem solver),
        which is specialized for 0-1 problems like this one. This is a
        targeted change to leverage a more specific, potentially faster,
        algorithm within the OR-Tools suite. Includes a greedy fallback.
        """
        value, demand, supply = problem
        num_items = len(value)
        num_resources = len(supply)

        if num_items == 0:
            return []

        # 1. HEURISTIC: Generate a good initial solution for the hint and as a fallback.
        densities = []
        for i in range(num_items):
            weighted_demand = sum(demand[i][r] / supply[r] for r in range(num_resources) if supply[r] > 0)
            if weighted_demand > 1e-9:
                density = value[i] / weighted_demand
            else:
                density = float('inf') if value[i] > 0 else 0.0
            densities.append((density, i))

        densities.sort(key=lambda item: item[0], reverse=True)
        
        initial_solution_values = [0] * num_items
        greedy_solution_indices = []
        current_demand = [0] * num_resources
        for _, i in densities:
            if all(current_demand[r] + demand[i][r] <= supply[r] for r in range(num_resources)):
                initial_solution_values[i] = 1
                greedy_solution_indices.append(i)
                for r in range(num_resources):
                    current_demand[r] += demand[i][r]

        # 2. MIP SOLVER SETUP (BOP)
        solver = pywraplp.Solver.CreateSolver('BOP')
        if not solver:
            return greedy_solution_indices # Fallback if solver creation fails

        x = [solver.IntVar(0, 1, f'x_{i}') for i in range(num_items)]
        solver.SetHint(x, initial_solution_values)

        for r in range(num_resources):
            solver.Add(solver.Sum(x[i] * demand[i][r] for i in range(num_items)) <= supply[r])
        
        solver.Maximize(solver.Sum(x[i] * value[i] for i in range(num_items)))

        # 3. SOLVE
        time_limit = kwargs.get('time_limit')
        if time_limit is not None:
            solver.set_time_limit(int(time_limit * 1000))
        
        status = solver.Solve()

        # 4. EXTRACT RESULTS
        if status in (pywraplp.Solver.OPTIMAL, pywraplp.Solver.FEASIBLE):
            solution_indices = []
            for i in range(num_items):
                if x[i].solution_value() > 0.5:
                    solution_indices.append(i)
            return solution_indices
        
        # If BOP fails (e.g., times out), the greedy solution is a valid fallback.
        return greedy_solution_indices