import numpy as np
from ortools.linear_solver import pywraplp
from ortools.sat.python import cp_model

class Solver:
    def solve(self, problem, **kwargs):
        """
        Solve the TSP problem using OR-Tools CP-SAT solver.
        
        :param problem: Distance matrix as a list of lists.
        :return: A list representing the optimal tour, starting and ending at city 0.
        """
        n = len(problem)
        
        if n <= 1:
            return [0, 0]
            
        # Use OR-Tools CP-SAT solver
        model = cp_model.CpModel()
        
        # Create variables using circuit constraint
        arcs = []
        x = {}
        for i in range(n):
            for j in range(n):
                if i != j:
                    var = model.NewBoolVar(f'x[{i},{j}]')
                    x[i, j] = var
                    arcs.append((i, j, var))
        
        # Add circuit constraint
        model.AddCircuit(arcs)
        
        # Add objective
        model.Minimize(sum(problem[i][j] * x[i, j] for i in range(n) for j in range(n) if i != j))
        
        # Solve
        solver = cp_model.CpSolver()
        solver.parameters.max_time_in_seconds = 10.0
        status = solver.Solve(model)
        
        if status in (cp_model.OPTIMAL, cp_model.FEASIBLE):
            # Extract solution by following the tour from city 0
            path = [0]
            current = 0
            
            # Follow the tour until we get back to city 0
            while len(path) < n:
                for j in range(n):
                    if current != j and solver.Value(x[current, j]) == 1:
                        path.append(j)
                        current = j
                        break
            
            path.append(0)  # Return to start
            return path
        else:
            # Fallback - this should not happen for valid inputs
            return [0] + list(range(1, n)) + [0]
        
        # Build tour using nearest neighbor starting from city 0
        unvisited = set(range(1, n))
        current = 0
        path = [0]
        
        while unvisited:
            nearest = min(unvisited, key=lambda x: distance_matrix[current][x])
            path.append(nearest)
            unvisited.remove(nearest)
            current = nearest
        
        path.append(0)  # Return to start
        
        # Apply 2-opt improvement
        path = self._two_opt(path, distance_matrix)
        
        return path
    
    def _fast_heuristic(self, distance_matrix):
        """Fast heuristic for larger instances."""
        return self._nearest_neighbor_simple(distance_matrix)
    
    def _nearest_neighbor_simple(self, distance_matrix):
        """Simple and fast nearest neighbor."""
        n = len(distance_matrix)
        if n <= 1:
            return [0, 0]
        
        unvisited = set(range(1, n))
        current = 0
        path = [0]
        
        while unvisited:
            nearest = min(unvisited, key=lambda x: distance_matrix[current][x])
            path.append(nearest)
            unvisited.remove(nearest)
            current = nearest
        
        path.append(0)
        return path
    
    def _two_opt(self, tour, distance_matrix):
        """2-opt improvement heuristic."""
        n = len(tour)
        improved = True
        
        while improved:
            improved = False
            for i in range(1, n - 2):
                for j in range(i + 1, n - 1):
                    # Calculate the change in distance if we reverse tour[i:j+1]
                    old_distance = (distance_matrix[tour[i-1]][tour[i]] + 
                                   distance_matrix[tour[j]][tour[j+1]])
                    new_distance = (distance_matrix[tour[i-1]][tour[j]] + 
                                   distance_matrix[tour[i]][tour[j+1]])
                    
                    if new_distance < old_distance:
                        # Reverse the segment
                        tour[i:j+1] = tour[i:j+1][::-1]
                        improved = True
        
        return tour
    
    def _calculate_tour_distance(self, tour, distance_matrix):
        """Calculate the total distance of a tour."""
        if len(tour) <= 1:
            return 0
        return sum(distance_matrix[tour[i]][tour[i + 1]] for i in range(len(tour) - 1))