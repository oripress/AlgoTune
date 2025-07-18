import numpy as np
from numba import jit
import pulp

class Solver:
    def solve(self, problem: list[list[int]]) -> list[int]:
        """
        Solves the minimum dominating set problem using a hybrid approach.
        
        :param problem: A 2d adjacency matrix representing the graph.
        :return: A list of node indices included in the minimum dominating set.
        """
        n = len(problem)
        if n == 0:
            return []
        
        # For small graphs, use exact ILP solver
        if n <= 50:
            return self._solve_ilp(problem)
        
        # For medium graphs, use a fast approximation with local search
        elif n <= 200:
            return self._solve_hybrid(problem)
        
        # For large graphs, use OR-Tools with optimizations
        else:
            return self._solve_ortools_optimized(problem)
    
    def _solve_ilp(self, problem):
        """Solve using Integer Linear Programming with PuLP."""
        n = len(problem)
        
        # Create the model
        model = pulp.LpProblem("MinimumDominatingSet", pulp.LpMinimize)
        
        # Decision variables
        x = [pulp.LpVariable(f"x_{i}", cat='Binary') for i in range(n)]
        
        # Objective: minimize the number of vertices in dominating set
        model += pulp.lpSum(x)
        
        # Constraints: each vertex must be dominated
        for i in range(n):
            # Vertex i is dominated if it's selected or at least one neighbor is selected
            neighbors = [x[i]]
            for j in range(n):
                if problem[i][j] == 1:
                    neighbors.append(x[j])
            model += pulp.lpSum(neighbors) >= 1
        
        # Solve
        model.solve(pulp.PULP_CBC_CMD(msg=0))
        
        # Extract solution
        if model.status == pulp.LpStatusOptimal:
            return [i for i in range(n) if x[i].value() > 0.5]
        else:
            # Fallback to OR-Tools
            return self._solve_ortools_optimized(problem)
    
    def _solve_hybrid(self, problem):
        """Hybrid approach for medium-sized graphs."""
        n = len(problem)
        adj = np.array(problem, dtype=np.int8)
        
        # Start with a greedy solution
        initial = self._greedy_solution(adj, n)
        
        # Try to improve using branch and bound
        best = self._branch_and_bound(adj, n, len(initial))
        
        if best is not None:
            return best
        else:
            # Fallback to ILP
            return self._solve_ilp(problem)
    
    def _solve_ortools_optimized(self, problem):
        """Optimized OR-Tools solver with hints and parameters."""
        from ortools.sat.python import cp_model
        
        n = len(problem)
        model = cp_model.CpModel()
        
        # Create boolean variables
        nodes = [model.NewBoolVar(f"x_{i}") for i in range(n)]
        
        # Add domination constraints
        for i in range(n):
            neighbors = [nodes[i]]
            for j in range(n):
                if problem[i][j] == 1:
                    neighbors.append(nodes[j])
            model.Add(sum(neighbors) >= 1)
        
        # Objective
        model.Minimize(sum(nodes))
        
        # Get initial greedy solution as hint
        adj = np.array(problem, dtype=np.int8)
        hint = self._greedy_solution(adj, n)
        for i in range(n):
            model.AddHint(nodes[i], 1 if i in hint else 0)
        
        # Solve with optimized parameters
        solver = cp_model.CpSolver()
        solver.parameters.num_search_workers = 4
        solver.parameters.max_time_in_seconds = 5.0
        status = solver.Solve(model)
        
        if status == cp_model.OPTIMAL:
            return [i for i in range(n) if solver.Value(nodes[i]) == 1]
        else:
            return []
    
    def _greedy_solution(self, adj, n):
        """Fast greedy algorithm to get initial solution."""
        dominated = np.zeros(n, dtype=bool)
        dominating_set = []
        
        while not np.all(dominated):
            best_vertex = -1
            best_count = 0
            
            for v in range(n):
                if v in dominating_set:
                    continue
                
                # Count new vertices that would be dominated
                count = 0
                if not dominated[v]:
                    count += 1
                
                for u in range(n):
                    if adj[v][u] == 1 and not dominated[u]:
                        count += 1
                
                if count > best_count:
                    best_count = count
                    best_vertex = v
            
            if best_vertex == -1:
                break
            
            dominating_set.append(best_vertex)
            dominated[best_vertex] = True
            for u in range(n):
                if adj[best_vertex][u] == 1:
                    dominated[u] = True
        
        return dominating_set
    
    def _branch_and_bound(self, adj, n, upper_bound):
        """Branch and bound algorithm for exact solution."""
        best_solution = None
        best_size = upper_bound
        
        def backtrack(current_set, dominated, remaining_vertices):
            nonlocal best_solution, best_size
            
            # Pruning
            if len(current_set) >= best_size:
                return
            
            # Check if all vertices are dominated
            if np.all(dominated):
                if len(current_set) < best_size:
                    best_size = len(current_set)
                    best_solution = list(current_set)
                return
            
            # If no remaining vertices, can't dominate all
            if not remaining_vertices:
                return
            
            # Choose next vertex to branch on
            # Pick vertex that dominates the most undominated vertices
            best_v = -1
            best_count = 0
            for v in remaining_vertices:
                count = 0
                if not dominated[v]:
                    count += 1
                for u in range(n):
                    if adj[v][u] == 1 and not dominated[u]:
                        count += 1
                if count > best_count:
                    best_count = count
                    best_v = v
            
            if best_v == -1:
                return
            
            # Branch 1: Include best_v
            new_dominated = dominated.copy()
            new_dominated[best_v] = True
            for u in range(n):
                if adj[best_v][u] == 1:
                    new_dominated[u] = True
            
            new_remaining = [v for v in remaining_vertices if v != best_v]
            backtrack(current_set + [best_v], new_dominated, new_remaining)
            
            # Branch 2: Exclude best_v (only if it's already dominated)
            if dominated[best_v]:
                backtrack(current_set, dominated, new_remaining)
        
        # Start branch and bound
        initial_dominated = np.zeros(n, dtype=bool)
        all_vertices = list(range(n))
        backtrack([], initial_dominated, all_vertices)
        
        return best_solution