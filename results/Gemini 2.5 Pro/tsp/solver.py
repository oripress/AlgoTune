from typing import Any, List
from ortools.sat.python import cp_model

class Solver:
    def solve(self, problem: List[List[int]], **kwargs: Any) -> List[int]:
        """
        Solves the TSP using an optimized CP-SAT model.
        This version adds redundant constraints and parallelizes the search
        to improve performance over the basic CP-SAT model.
        """
        n = len(problem)
        if n == 0:
            return []
        if n == 1:
            return [0, 0]

        # Create the CP-SAT model.
        model = cp_model.CpModel()

        # Create the literal variables for the arcs in the tour.
        # x[i, j] is true if the tour goes from node i to node j.
        x = {}
        for i in range(n):
            for j in range(n):
                if i != j:
                    x[i, j] = model.NewBoolVar(f'x_{i}_{j}')

        # Add redundant constraints: each node must have exactly one incoming and one outgoing arc.
        # While AddCircuit implies this, explicitly adding them can help the solver.
        for i in range(n):
            model.AddExactlyOne(x[i, j] for j in range(n) if i != j)  # Outgoing
            model.AddExactlyOne(x[j, i] for j in range(n) if i != j)  # Incoming

        # Add the main circuit constraint.
        arcs = []
        for i in range(n):
            for j in range(n):
                if i != j:
                    arcs.append((i, j, x[i, j]))
        model.AddCircuit(arcs)

        # Define the objective function: minimize the total tour cost.
        total_cost = sum(problem[i][j] * x[i, j] for i, j in x)
        model.Minimize(total_cost)

        # Create a solver and configure for performance.
        solver = cp_model.CpSolver()

        # Use multiple workers to parallelize the search.
        solver.parameters.num_search_workers = 8

        # Set a time limit if provided to avoid timeouts.
        time_limit = kwargs.get('time_limit')
        if time_limit is not None:
            solver.parameters.max_time_in_seconds = float(time_limit)

        status = solver.Solve(model)

        # Reconstruct the path if a solution is found.
        if status == cp_model.OPTIMAL or status == cp_model.FEASIBLE:
            path = [0]
            current_city = 0
            for _ in range(n - 1):
                for next_city in range(n):
                    if current_city != next_city and solver.Value(x[current_city, next_city]):
                        path.append(next_city)
                        current_city = next_city
                        break
            
            path.append(0)  # Complete the tour by returning to the start.
            return path
        else:
            return []