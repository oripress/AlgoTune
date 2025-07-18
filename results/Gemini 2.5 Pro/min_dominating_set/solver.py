import numpy as np
from ortools.sat.python import cp_model
from scipy.sparse.csgraph import connected_components
from scipy.sparse import csr_matrix
from typing import Any

class Solver:
    def solve(self, problem: list[list[int]], **kwargs) -> Any:
        n = len(problem)
        if n == 0:
            return []

        problem_np = np.array(problem, dtype=np.int8)

        # Step 1: Decompose into connected components. This is a massive optimization.
        if n > 1:
            graph_sparse = csr_matrix(problem_np)
            n_components, labels = connected_components(
                csgraph=graph_sparse, directed=False, return_labels=True
            )

            if n_components > 1:
                final_solution = []
                for i in range(n_components):
                    component_nodes_mask = labels == i
                    component_nodes = np.where(component_nodes_mask)[0]
                    sub_adj_matrix = problem_np[component_nodes_mask][:, component_nodes_mask]
                    # Solve subproblem recursively
                    sub_solution_local = self.solve(sub_adj_matrix.tolist())
                    # Map local indices back to original graph indices
                    for local_idx in sub_solution_local:
                        final_solution.append(component_nodes[local_idx])
                return sorted(final_solution)

        # If we reach here, the graph is connected (or n <= 1).
        # Base cases for recursion or small graphs.
        if n <= 1:
            return list(range(n))

        degrees = problem_np.sum(axis=1)
        
        # Rule: Universal vertex (dominating set of size 1)
        if np.any(degrees == n - 1):
            return [int(np.argmax(degrees))]

        solution = set()
        
        # Rule: Leaf nodes (vectorized)
        leaf_indices = np.where(degrees == 1)[0]
        if leaf_indices.size > 0:
            support_indices = np.argmax(problem_np[leaf_indices], axis=1)
            solution.update(support_indices.tolist())

        # Rule: N[u] subset N[v] (dominance rule) - fully vectorized
        closed_adj_matrix = problem_np + np.identity(n, dtype=np.int8)
        
        subset_matrix = np.all(closed_adj_matrix[:, np.newaxis, :] <= closed_adj_matrix[np.newaxis, :, :], axis=2)
        
        deg_lt_mask = degrees[:, np.newaxis] < degrees[np.newaxis, :]
        deg_eq_mask = degrees[:, np.newaxis] == degrees[np.newaxis, :]
        idx_gt_mask = np.arange(n)[:, np.newaxis] > np.arange(n)[np.newaxis, :]
        tie_break_mask = deg_eq_mask & idx_gt_mask
        
        dominance_cond_mask = deg_lt_mask | tie_break_mask
        dominance_matrix = subset_matrix & dominance_cond_mask
        
        is_excludable = np.any(dominance_matrix, axis=1)
        
        nodes_in_solution_mask = np.zeros(n, dtype=bool)
        if solution:
            nodes_in_solution_mask[list(solution)] = True
            
        final_exclusion_mask = is_excludable & ~nodes_in_solution_mask
        nodes_to_exclude = set(np.where(final_exclusion_mask)[0])

        # --- Subproblem construction ---
        is_dominated = np.zeros(n, dtype=bool)
        if solution:
            sol_nodes = list(solution)
            is_dominated[sol_nodes] = True
            dominated_by_neighbors = np.any(problem_np[sol_nodes, :], axis=0)
            is_dominated |= dominated_by_neighbors
        
        if np.all(is_dominated):
            return sorted(list(solution))

        nodes_to_dominate = np.where(~is_dominated)[0]
        
        candidate_nodes_mask = np.ones(n, dtype=bool)
        candidate_nodes_mask[list(solution.union(nodes_to_exclude))] = False
        candidate_nodes = np.where(candidate_nodes_mask)[0]

        if not nodes_to_dominate.size or not candidate_nodes.size:
            return sorted(list(solution))
        
        node_map = {node: i for i, node in enumerate(candidate_nodes)}
        
        model = cp_model.CpModel()
        sub_nodes = [model.NewBoolVar(f"x_{i}") for i in range(len(candidate_nodes))]
        model.Minimize(sum(sub_nodes))
        
        for i_orig in nodes_to_dominate:
            dominators = np.union1d(np.array([i_orig]), np.where(problem_np[i_orig] == 1)[0])
            constraint_vars = [sub_nodes[node_map[d]] for d in dominators if d in node_map]
            if constraint_vars:
                model.Add(sum(constraint_vars) >= 1)
        
        solver = cp_model.CpSolver()
        status = solver.Solve(model)
        
        sub_solution = set()
        if status == cp_model.OPTIMAL or status == cp_model.FEASIBLE:
            sub_solution = {candidate_nodes[i] for i, node_var in enumerate(sub_nodes) if solver.Value(node_var) == 1}
        
        return sorted(list(solution.union(sub_solution)))