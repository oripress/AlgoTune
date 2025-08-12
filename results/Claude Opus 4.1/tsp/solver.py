import numpy as np
from ortools.sat.python import cp_model

class Solver:
    def solve(self, problem, **kwargs):
        """
        Solve the TSP problem using optimized CP-SAT solver.
        
        :param problem: Distance matrix as a list of lists.
        :return: A list representing the optimal tour, starting and ending at city 0.
        """
        n = len(problem)
        
        if n <= 1:
            return [0, 0]
        
        if n <= 2:
            return list(range(n)) + [0]
        
        # Use CP-SAT solver for optimal solution
        model = cp_model.CpModel()
        
        # Create variables - only for edges we might use
        x = {}
        for i in range(n):
            for j in range(n):
                if i != j:
                    x[i, j] = model.NewBoolVar(f'x[{i},{j}]')
        
        # Circuit constraint ensures we visit each city exactly once
        arcs = []
        for i in range(n):
            for j in range(n):
                if i != j:
                    arcs.append((i, j, x[i, j]))
        model.AddCircuit(arcs)
        
        # Minimize total distance
        objective = []
        for i in range(n):
            for j in range(n):
                if i != j:
                    objective.append(problem[i][j] * x[i, j])
        model.Minimize(sum(objective))
        
        # Solve with optimized parameters
        solver = cp_model.CpSolver()
        solver.parameters.num_search_workers = 1  # Single thread is often faster for small problems
        solver.parameters.log_search_progress = False  # Disable logging for speed
        
        # For small problems, use aggressive time limits and heuristics
        if n <= 10:
            solver.parameters.max_time_in_seconds = 0.1
        elif n <= 15:
            solver.parameters.max_time_in_seconds = 0.5
        else:
            solver.parameters.max_time_in_seconds = 2.0
        
        status = solver.Solve(model)
        
        if status in (cp_model.OPTIMAL, cp_model.FEASIBLE):
            # Reconstruct path
            path = []
            current = 0
            visited = set()
            
            while len(visited) < n:
                path.append(current)
                visited.add(current)
                
                # Find next city
                for next_city in range(n):
                    if current != next_city and solver.Value(x[current, next_city]) == 1:
                        current = next_city
                        break
            
            path.append(0)  # Return to start
            return path
        else:
            # Fallback to nearest neighbor if solver fails
            return self._nearest_neighbor(problem, n)
    
    def _nearest_neighbor(self, problem, n):
        """Nearest neighbor heuristic as fallback"""
        path = [0]
        visited = set([0])
        current = 0
        
        while len(visited) < n:
            best_dist = float('inf')
            best_city = -1
            
            for next_city in range(n):
                if next_city not in visited:
                    if problem[current][next_city] < best_dist:
                        best_dist = problem[current][next_city]
                        best_city = next_city
            
            if best_city != -1:
                path.append(best_city)
                visited.add(best_city)
                current = best_city
        
        path.append(0)
        return path