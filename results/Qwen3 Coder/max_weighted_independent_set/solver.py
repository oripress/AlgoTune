import numpy as np
from ortools.sat.python import cp_model
import networkx as nx
class Solver:
    def solve(self, problem, **kwargs):
        """Solve Maximum Weighted Independent Set problem"""
        adj_matrix = problem["adj_matrix"]
        weights = problem["weights"]
        n = len(adj_matrix)
        
        # For very small graphs, use brute force
        if n <= 8:
            return self._solve_networkx(adj_matrix, weights)
        # For medium graphs, use greedy with local search
        elif n <= 25:
            return self._solve_local_search(adj_matrix, weights)
        # For larger graphs, use exact solver
        return self._solve_exact(adj_matrix, weights)
    
    
    def _is_independent_set(self, adj_matrix, solution):
        """Check if a solution is an independent set"""
        for a in range(len(solution)):
            for b in range(a + 1, len(solution)):
                if adj_matrix[solution[a]][solution[b]]:
                    return False
        return True
    
    def _solve_exact(self, adj_matrix, weights):
        """Solve using CP-SAT for optimal solution"""
        n = len(adj_matrix)
        model = cp_model.CpModel()
        nodes = [model.NewBoolVar(f"x_{i}") for i in range(n)]

        # Add constraints: no two adjacent nodes can both be selected
        for i in range(n):
            for j in range(i + 1, n):
                if adj_matrix[i][j]:
                    model.Add(nodes[i] + nodes[j] <= 1)

        # Objective: maximize total weight
        model.Maximize(sum(weights[i] * nodes[i] for i in range(n)))

        # Solve with optimized parameters
        solver = cp_model.CpSolver()
        solver.parameters.max_time_in_seconds = 3.0
        solver.parameters.num_search_workers = 2  # Use fewer threads for better performance
        solver.parameters.log_search_progress = False
        solver.parameters.cp_model_probing_level = 0  # Disable probing for speed
        status = solver.Solve(model)
        
        if status in [cp_model.OPTIMAL, cp_model.FEASIBLE]:
            return [i for i in range(n) if solver.Value(nodes[i])]
        else:
            return []
    
    def _solve_networkx(self, adj_matrix, weights):
        """Solve using networkx for small graphs"""
        # For tiny graphs, we can find maximum weighted independent set
        # by checking all maximal independent sets (this is still exponential
        # but faster for very small graphs)
        n = len(adj_matrix)
        
        if n <= 10:
            # Brute force for tiny graphs
            max_weight = -1
            best_set = []
            
            # Generate all subsets and check if they're independent sets
            for i in range(1 << n):
                nodes = [j for j in range(n) if (i & (1 << j))]
                if self._is_independent_set(adj_matrix, nodes):
                    weight = sum(weights[j] for j in nodes)
                    if weight > max_weight:
                        max_weight = weight
                        best_set = nodes
            return sorted(best_set)
        else:
            # For slightly larger graphs, use greedy approximation with local search
            greedy_solution = self._solve_greedy(adj_matrix, weights)
            return self._solve_local_search(adj_matrix, weights)
    
    def _solve_greedy(self, adj_matrix, weights):
        """Greedy approximation for larger problems"""
        n = len(adj_matrix)
        
        # Create adjacency list representation
        adj_list = [[] for _ in range(n)]
        for i in range(n):
            for j in range(i + 1, n):
                if adj_matrix[i][j] == 1:
                    adj_list[i].append(j)
                    adj_list[j].append(i)
        
        # Greedy algorithm: sort nodes by weight and select if independent
        # Use weight/degree ratio for better selection
        node_ratios = []
        for i in range(n):
            degree = len(adj_list[i])
            ratio = weights[i] / (degree + 1)  # Add 1 to avoid division by zero
            node_ratios.append((ratio, weights[i], i))
        
        node_ratios.sort(reverse=True)  # Sort by ratio descending
        selected = []
        selected_mask = [False] * n
        blocked_mask = [False] * n  # Nodes that can't be selected due to conflicts

        for ratio, weight, node in node_ratios:
            # Check if node can be selected
            if not blocked_mask[node]:
                selected.append(node)
                selected_mask[node] = True
                # Block all neighbors
                for neighbor in adj_list[node]:
                    blocked_mask[neighbor] = True

        return sorted(selected)
    def _solve_local_search(self, adj_matrix, weights):
        """Local search improvement for greedy solution"""
        # Start with greedy solution
        current_solution = self._solve_greedy(adj_matrix, weights)

        # Try to improve by adding/removing nodes
        n = len(adj_matrix)
        current_weight = sum(weights[i] for i in current_solution)
        solution_set = set(current_solution)

        # Simple local search: try adding each non-selected node
        for i in range(n):
            if i not in solution_set:
                # Check if we can add this node
                can_add = True
                for j in range(n):
                    if adj_matrix[i][j] == 1 and j in solution_set:
                        can_add = False
                        break

                if can_add:
                    # Try adding this node
                    new_solution = list(solution_set)
                    new_solution.append(i)
                    new_weight = sum(weights[j] for j in new_solution)

                    if new_weight > current_weight:
                        current_solution = sorted(new_solution)
                        solution_set = set(current_solution)
                        current_weight = new_weight

        return current_solution