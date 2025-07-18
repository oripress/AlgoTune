from typing import Any, Set, List

class Solver:
    def solve(self, problem: list[list[int]], **kwargs) -> Any:
        """
        This solution abandons external libraries, which have proven incompatible
        with the execution environment, and instead implements a highly optimized,
        pure-Python maximum clique algorithm from scratch. This is the ultimate
        "outside the box" strategy to bypass library overhead and environment issues.

        The implementation is based on the Bron-Kerbosch algorithm with two key
        optimizations:
        1.  **Tomita Pivoting**: A sophisticated pivot selection rule to prune the
            search tree aggressively.
        2.  **Maximum Clique Pruning**: The search is tailored to find only the
            *maximum* clique, not all maximal cliques. It tracks the size of the
            largest clique found so far and abandons branches that cannot surpass it.

        The graph is represented as a lightweight adjacency list (dict of sets)
        to minimize overhead compared to networkx.Graph objects.
        """
        n = len(problem)
        if n == 0:
            return []

        # 1. Build a lightweight adjacency list (dict of sets)
        adj = {i: set() for i in range(n)}
        has_edges = False
        for i in range(n):
            for j in range(i + 1, n):
                if problem[i][j] == 1:
                    adj[i].add(j)
                    adj[j].add(i)
                    has_edges = True
        
        if not has_edges:
            return [0] if n > 0 else []

        # State for the recursive solver
        self.max_clique: List[int] = []
        self.adj = adj
        
        # 2. Initial call to the recursive solver
        self.find_max_clique_recursive(set(), set(adj.keys()), set())
        
        return sorted(self.max_clique)

    def find_max_clique_recursive(self, R: Set[int], P: Set[int], X: Set[int]):
        """
        Bron-Kerbosch algorithm with pivoting and max-clique pruning.
        R: The nodes in the current clique.
        P: Candidate nodes that can extend the clique.
        X: Nodes already processed, not to be used again.
        """
        # Pruning Step: If the current clique plus remaining candidates can't
        # beat the best found so far, stop.
        if len(R) + len(P) <= len(self.max_clique):
            return

        if not P and not X:
            # Base Case: A maximal clique is found. Check if it's the new maximum.
            if len(R) > len(self.max_clique):
                self.max_clique = list(R)
            return

        if not P:
            return

        # Pivot Selection (Tomita's strategy): Choose a pivot `u` in P U X
        # that maximizes the number of its neighbors in P.
        try:
            pivot = max(P | X, key=lambda u: len(P & self.adj[u]))
        except ValueError:
            # This can happen if P | X is empty, though guarded by checks above.
            return

        # Iterate through candidates that are NOT neighbors of the pivot.
        # This is the core of the pruning strategy.
        P_without_pivot_neighbors = P - self.adj[pivot]
        
        # Iterate on a copy as we modify P in the loop
        for v in list(P_without_pivot_neighbors):
            self.find_max_clique_recursive(
                R | {v},
                P & self.adj[v],
                X & self.adj[v]
            )
            # Move v from candidates to excluded
            P.remove(v)
            X.add(v)