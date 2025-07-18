import numpy as np
from ortools.sat.python import cp_model
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components
import concurrent.futures
import os
from collections import deque

class Solver:
    def solve(self, problem, **kwargs):
        adj_matrix = np.array(problem["adj_matrix"])
        weights = np.array(problem["weights"])
        n = len(adj_matrix)
        
        # Preprocessing: remove isolated nodes with non-positive weight
        degrees = np.sum(adj_matrix, axis=1)
        isolated = (degrees == 0)
        reduced_indices = np.arange(n)
        mask = np.ones(n, dtype=bool)
        
        for i in range(n):
            if isolated[i] and weights[i] <= 0:
                mask[i] = False
        
        # Create reduced graph
        adj_reduced = adj_matrix[mask][:, mask]
        weights_reduced = weights[mask]
        reduced_indices = reduced_indices[mask]
        n_reduced = len(weights_reduced)
        
        if n_reduced == 0:
            return []
        
        # Find connected components
        graph = csr_matrix(adj_reduced)
        n_components, labels = connected_components(
            csgraph=graph, directed=False, return_labels=True
        )
        
        if n_components == 0:
            return []
        
        # Use parallel execution for components
        solution = []
        max_workers = min(8, os.cpu_count() or 1)
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = []
            for comp_id in range(n_components):
                comp_nodes = np.where(labels == comp_id)[0]
                comp_size = len(comp_nodes)
                if comp_size == 0:
                    continue
                comp_weights = weights_reduced[comp_nodes]
                comp_adj = adj_reduced[comp_nodes][:, comp_nodes]
                
                # Apply graph reduction rules
                forced, remaining = self._reduce_component(comp_adj, comp_weights)
                solution.extend([int(reduced_indices[comp_nodes[i]]) for i in forced])
                
                # Skip if no nodes left after reduction
                if not remaining:
                    continue
                
                # Create new component from remaining nodes
                new_comp_nodes = comp_nodes[remaining]
                new_comp_size = len(new_comp_nodes)
                new_comp_weights = weights_reduced[new_comp_nodes]
                new_comp_adj = adj_reduced[new_comp_nodes][:, new_comp_nodes]
                
                # Submit the component for solving
                futures.append(executor.submit(
                    self._solve_component, 
                    new_comp_nodes, 
                    new_comp_adj, 
                    new_comp_weights, 
                    reduced_indices
                ))
            
            for future in concurrent.futures.as_completed(futures):
                comp_solution = future.result()
                solution.extend(comp_solution)
        
        return solution

    def _reduce_component(self, adj_matrix, weights):
        """Apply graph reduction rules to a component"""
        n = len(weights)
        if n == 0:
            return [], []
        
        # Initialize data structures
        degree = np.sum(adj_matrix, axis=1)
        remaining = set(range(n))
        forced = []  # Nodes forced into solution
        changes = True
        
        while changes:
            changes = False
            # Process nodes in order
            nodes = list(remaining)
            for i in nodes:
                if i not in remaining:
                    continue
                    
                # Rule 1: Isolated nodes (degree 0)
                if degree[i] == 0:
                    if weights[i] > 0:
                        forced.append(i)
                    remaining.remove(i)
                    changes = True
                    continue
                    
                # Rule 2: Pendant vertices (degree 1)
                if degree[i] == 1:
                    # Find the neighbor
                    neighbor = None
                    for j in range(n):
                        if adj_matrix[i, j] and j in remaining:
                            neighbor = j
                            break
                            
                    if neighbor is None:
                        # Shouldn't happen, but handle safely
                        if weights[i] > 0:
                            forced.append(i)
                        remaining.remove(i)
                        changes = True
                        continue
                        
                    # Compare weights
                    if weights[i] > weights[neighbor]:
                        forced.append(i)
                        remaining.remove(i)
                        remaining.remove(neighbor)
                        # Update degrees for neighbor's neighbors
                        for k in range(n):
                            if adj_matrix[neighbor, k] and k in remaining:
                                degree[k] -= 1
                        changes = True
                    else:
                        forced.append(neighbor)
                        remaining.remove(i)
                        remaining.remove(neighbor)
                        # Update degrees for neighbor's neighbors
                        for k in range(n):
                            if adj_matrix[neighbor, k] and k in remaining:
                                degree[k] -= 1
                        changes = True
                    continue
                    
                # Rule 3: Twin reduction (non-adjacent with same neighborhood)
                for j in range(i+1, n):
                    if j not in remaining or adj_matrix[i, j]:
                        continue
                        
                    # Check if same neighborhood
                    same_neighbors = True
                    for k in range(n):
                        if k == i or k == j:
                            continue
                        if adj_matrix[i, k] != adj_matrix[j, k]:
                            same_neighbors = False
                            break
                            
                    if same_neighbors:
                        # Twin nodes found
                        if weights[i] >= weights[j]:
                            # Keep i, remove j
                            remaining.remove(j)
                            # Update degrees for j's neighbors
                            for k in range(n):
                                if adj_matrix[j, k] and k in remaining:
                                    degree[k] -= 1
                        else:
                            # Keep j, remove i
                            remaining.remove(i)
                            # Update degrees for i's neighbors
                            for k in range(n):
                                if adj_matrix[i, k] and k in remaining:
                                    degree[k] -= 1
                            break  # Break inner loop since i is removed
                        changes = True
        
        return forced, list(remaining)

    def _solve_component(self, comp_nodes, adj_reduced, weights_reduced, reduced_indices):
        comp_size = len(comp_nodes)
        
        if comp_size == 0:
            return []
        if comp_size == 1:
            if weights_reduced[0] > 0:
                return [int(reduced_indices[comp_nodes[0]])]
            return []
        
        # For small components (<=20 nodes), use optimized bitmask DP
        if comp_size <= 20:
            # Precompute neighbor masks
            nbr_mask = [0] * comp_size
            for i in range(comp_size):
                for j in range(comp_size):
                    if adj_reduced[i, j] and i != j:
                        nbr_mask[i] |= (1 << j)
            
            # DP initialization
            dp = [-10**18] * (1 << comp_size)
            dp[0] = 0
            best_from = [-1] * (1 << comp_size)
            taken_node = [-1] * (1 << comp_size)  # Track which node was taken
            
            # Iterate over masks in increasing order
            for mask in range(1 << comp_size):
                if dp[mask] < 0:
                    continue
                    
                # Find next node not in mask
                for i in range(comp_size):
                    if mask & (1 << i):
                        continue
                    
                    # Option 1: skip node i
                    new_mask = mask | (1 << i)
                    if dp[new_mask] < dp[mask]:
                        dp[new_mask] = dp[mask]
                        best_from[new_mask] = mask
                        taken_node[new_mask] = -1  # -1 means we skipped
                    
                    # Option 2: take node i and exclude neighbors
                    new_mask2 = mask | (1 << i) | nbr_mask[i]
                    new_val = dp[mask] + weights_reduced[i]
                    if new_val > dp[new_mask2]:
                        dp[new_mask2] = new_val
                        best_from[new_mask2] = mask
                        taken_node[new_mask2] = i  # Record taken node
            
            # Reconstruct solution using the taken_node array
            best_mask = max(range(1 << comp_size), key=lambda m: dp[m])
            current = best_mask
            selected = set()
            while current:
                if taken_node[current] != -1:
                    selected.add(taken_node[current])
                current = best_from[current]
            
            # Convert to original indices
            comp_solution = []
            for i in selected:
                comp_solution.append(int(reduced_indices[comp_nodes[i]]))
            return comp_solution
        
        # For larger components, use optimized CP-SAT
        model = cp_model.CpModel()
        x = [model.NewBoolVar(f'x_{i}') for i in range(comp_size)]
        
        # Add edge constraints
        for i in range(comp_size):
            for j in range(i+1, comp_size):
                if adj_reduced[i, j]:
                    model.Add(x[i] + x[j] <= 1)
        
        model.Maximize(sum(weights_reduced[i] * x[i] for i in range(comp_size)))
        
        solver = cp_model.CpSolver()
        # Optimized parameters
        solver.parameters.num_search_workers = 1  # Avoid oversubscription
        solver.parameters.log_search_progress = False
        solver.parameters.linearization_level = 1
        solver.parameters.symmetry_level = 0
        solver.parameters.max_time_in_seconds = 10.0  # Timeout for safety
        status = solver.Solve(model)
        
        comp_solution = []
        if status == cp_model.OPTIMAL or status == cp_model.FEASIBLE:
            for i in range(comp_size):
                if solver.Value(x[i]):
                    comp_solution.append(int(reduced_indices[comp_nodes[i]]))
        return comp_solution