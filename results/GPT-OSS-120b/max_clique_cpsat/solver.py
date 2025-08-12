import sys
sys.setrecursionlimit(10000)
from typing import List

class Solver:
    def solve(self, problem: List[List[int]], **kwargs) -> List[int]:
        """
        Exact maximum clique solver using a simple branch‑and‑bound
        Bron–Kerbosch algorithm with bit‑set representation.
        Returns a list of vertex indices (0‑based) forming a maximum clique.
        """
        n = len(problem)
        if n == 0:
            return []

        # Build adjacency bit masks (exclude self‑loops)
        adj: List[int] = [0] * n
        for i in range(n):
            mask = 0
            row = problem[i]
            for j, val in enumerate(row):
                if val and i != j:
                    mask |= 1 << j
            adj[i] = mask

        best_clique: List[int] = []

        def popcount(x: int) -> int:
            return x.bit_count()

        # Recursive expansion
        def expand(R: List[int], P: int) -> None:
            nonlocal best_clique
            # If no candidates left, R is a maximal clique
            if P == 0:
                if len(R) > len(best_clique):
                    best_clique = R.copy()
                return

            # Prune if even taking all remaining vertices cannot beat current best
            if len(R) + popcount(P) <= len(best_clique):
                return

            # Choose vertices from P one by one
            while P:
                # Upper bound check again (since P shrinks)
                if len(R) + popcount(P) <= len(best_clique):
                    return
                # Extract a vertex v (least‑significant set bit)
                v = (P & -P).bit_length() - 1
                # Branch: include v
                R.append(v)
                expand(R, P & adj[v])
                R.pop()
                # Remove v from candidate set
                P &= ~(1 << v)

        # Start with all vertices as candidates
        expand([], (1 << n) - 1)
        return best_clique