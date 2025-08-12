import numpy as np
from ortools.sat.python import cp_model
import networkx as nx
from itertools import combinations

class Solver:
    def solve(self, problem):
        """Solve the maximum common subgraph problem using advanced optimization techniques."""
        A = np.array(problem["A"])
        B = np.array(problem["B"])
        n, m = len(A), len(B)
        
        # Early termination for trivial cases
        if n == 0 or m == 0:
            return []
        
        # Convert to NetworkX graphs for advanced analysis
        G = nx.from_numpy_array(A)
        H = nx.from_numpy_array(B)
        
        # Pre-filter based on degree sequences and more sophisticated compatibility
        compatible_nodes = {}
        degrees_G = [G.degree(i) for i in range(n)]
        degrees_H = [H.degree(p) for p in range(m)]
        
        # More aggressive filtering based on structural properties
        for i in range(n):
            compatible_nodes[i] = []
            for p in range(m):
                # Check degree compatibility with more tolerance
                if abs(degrees_G[i] - degrees_H[p]) <= max(2, degrees_G[i] * 0.5):
                    # Additional filtering based on neighbor degrees
                    neighbors_G = list(G.neighbors(i))
                    neighbors_H = list(H.neighbors(p))
                    
                    # Check if the number of neighbors is compatible
                    if abs(len(neighbors_G) - len(neighbors_H)) <= 2:
                        compatible_nodes[i].append(p)
        
        # Use OR-Tools for constraint satisfaction with optimized settings
        model = cp_model.CpModel()
        
        # x[i][p] = 1 if node i in G is mapped to node p in H
        x = [[model.NewBoolVar(f"x_{i}_{p}") for p in range(m)] for i in range(n)]
        
        # Add constraints to enforce compatibility
        for i in range(n):
            for p in range(m):
                if p not in compatible_nodes[i]:
                    model.Add(x[i][p] == 0)
        
        # One-to-one mapping constraints
        for i in range(n):
            model.Add(sum(x[i][p] for p in range(m)) <= 1)
        for p in range(m):
            model.Add(sum(x[i][p] for i in range(n)) <= 1)
        
        # Edge consistency constraints - highly optimized version
        # Precompute all inconsistent pairs to reduce constraint creation
        inconsistent_pairs = []
        for i in range(n):
            for j in range(i + 1, n):
                if A[i, j] == 1:  # Edge exists in G
                    for p in compatible_nodes[i]:
                        for q in compatible_nodes[j]:
                            if p != q and B[p, q] == 0:  # No edge in H
                                inconsistent_pairs.append((i, j, p, q))
                else:  # No edge in G
                    for p in compatible_nodes[i]:
                        for q in compatible_nodes[j]:
                            if p != q and B[p, q] == 1:  # Edge exists in H
                                inconsistent_pairs.append((i, j, p, q))
        
        # Add all constraints at once for better performance
        for i, j, p, q in inconsistent_pairs:
            model.Add(x[i][p] + x[j][q] <= 1)
        
        # Objective: maximize size of the mapping
        model.Maximize(sum(x[i][p] for i in range(n) for p in range(m)))
        
        # Configure solver for maximum speed
        solver = cp_model.CpSolver()
        solver.parameters.max_time_in_seconds = 10.0
        solver.parameters.num_search_workers = 4  # Use multiple cores
        solver.parameters.log_search_progress = False
        solver.parameters.cp_model_probing_level = 0  # Reduce probing
        solver.parameters.cp_model_presolve = True
        solver.parameters.search_branching = cp_model.FIXED_SEARCH  # Use fixed search
        solver.parameters.linearization_level = 0  # Reduce linearization
        solver.parameters.optimize_with_core = True  # Use core-based optimization
        
        status = solver.Solve(model)
        
        if status in [cp_model.OPTIMAL, cp_model.FEASIBLE]:
            result = []
            for i in range(n):
                for p in range(m):
                    if solver.Value(x[i][p]) == 1:
                        result.append((i, p))
            return result
        else:
            return []