import sys
try:
    import clique_solver
except ImportError:
    pass

class Solver:
    def solve(self, problem: list[list[int]], **kwargs) -> list[int]:
        """
        Solves the maximum clique problem using the Bron-Kerbosch algorithm 
        implemented in Cython with bitsets and vertex ordering.
        """
        n = len(problem)
        if n == 0:
            return []

        # Degree ordering: Sort vertices by degree descending
        degrees = [sum(row) for row in problem]
        nodes_sorted = sorted(range(n), key=lambda i: degrees[i], reverse=True)
        
        old_to_new = {node: i for i, node in enumerate(nodes_sorted)}
        new_to_old = nodes_sorted
        
        # Reorder the adjacency matrix
        # We need to pass a list of lists of ints to the Cython function
        # The Cython function expects vector<vector<int>>
        
        # Create reordered matrix
        reordered_matrix = [[0] * n for _ in range(n)]
        for r in range(n):
            original_r = new_to_old[r]
            for c in range(n):
                if problem[original_r][new_to_old[c]]:
                    reordered_matrix[r][c] = 1
        
        # Call Cython solver
        # Note: The Cython function signature is solve_clique(int n, vector[vector[int]] matrix)
        # Python list of lists automatically converts to vector<vector<int>>
        
        solution_indices = clique_solver.solve_clique(n, reordered_matrix)
        
        # Map back to original indices
        original_indices = [new_to_old[i] for i in solution_indices]
        return sorted(original_indices)