import itertools
from typing import Any
from ortools.sat.python import cp_model

class Solver:
    def solve(self, problem: list[list[float]], **kwargs) -> Any:
        n = len(problem)
        if n <= 1:
            return [0, 0]
            
        import numpy as np
        import networkx as nx
        from scipy.sparse import csr_matrix
        from scipy.sparse.csgraph import maximum_bipartite_matching
        
        # Extract all unique weights
        unique_weights = set()
        for i in range(n):
            for j in range(i + 1, n):
                unique_weights.add(problem[i][j])
        
        sorted_weights = sorted(list(unique_weights))
        
        # Binary search for the minimum weight that allows a 2-factor
        left = 0
        right = len(sorted_weights) - 1
        min_idx = right
        
        while left <= right:
            mid = (left + right) // 2
            w = sorted_weights[mid]
            
            row_ind = []
            col_ind = []
            for i in range(n):
                for j in range(n):
                    if i != j and problem[i][j] <= w:
                        row_ind.append(i)
                        col_ind.append(j)
            
            if len(row_ind) > 0:
                data = np.ones(len(row_ind), dtype=int)
                graph = csr_matrix((data, (row_ind, col_ind)), shape=(n, n))
                matching = maximum_bipartite_matching(graph, perm_type='column')
                has_2_factor = np.all(matching >= 0)
            else:
                has_2_factor = False
                
            if has_2_factor:
                min_idx = mid
                right = mid - 1
            else:
                left = mid + 1
                
        def has_hamiltonian_cycle(max_w):
            # Fast graph checks
            G = nx.Graph()
            G.add_nodes_from(range(n))
            edges_to_add = []
            degrees = [0] * n
            for i in range(n):
                for j in range(i + 1, n):
                    if problem[i][j] <= max_w:
                        edges_to_add.append((i, j))
                        degrees[i] += 1
                        degrees[j] += 1
            
            if any(d < 2 for d in degrees):
                return False, []
                
            G.add_edges_from(edges_to_add)
            if not nx.is_biconnected(G):
                return False, []
                
            # CP-SAT model
            model = cp_model.CpModel()
            edges = []
            
            for i in range(n):
                for j in range(n):
                    if i != j and problem[i][j] <= max_w:
                        lit = model.NewBoolVar(f'edge_{i}_{j}')
                        edges.append((i, j, lit))
            
            model.AddCircuit(edges)
            
            solver = cp_model.CpSolver()
            solver.parameters.num_search_workers = 1
            status = solver.Solve(model)
            
            if status == cp_model.OPTIMAL or status == cp_model.FEASIBLE:
                next_node = {}
                for i, j, lit in edges:
                    if solver.BooleanValue(lit):
                        next_node[i] = j
                
                tour = [0]
                curr = 0
                for _ in range(n):
                    curr = next_node[curr]
                    tour.append(curr)
                return True, tour
            return False, []

        # Linear search upwards from the 2-factor lower bound
        for idx in range(min_idx, len(sorted_weights)):
            w = sorted_weights[idx]
            possible, tour = has_hamiltonian_cycle(w)
            if possible:
                return tour
                
        return []