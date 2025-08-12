from typing import List, Tuple, Dict, Any

class Solver:
    def solve(self, problem: Dict[str, List[List[int]]]) -> List[Tuple[int, int]]:
        """
        Find a maximum common induced subgraph between two undirected graphs
        given by adjacency matrices A and B.
        Returns a list of (node_in_A, node_in_B) pairs.
        """
        A = problem["A"]
        B = problem["B"]
        n, m = len(A), len(B)

        # Ensure we always map from the smaller graph to the larger one
        swapped = False
        if n > m:
            # swap roles of A and B
            A, B = B, A
            n, m = m, n
            swapped = True

        # Precompute degrees
        degA = [sum(row) for row in A]
        degB = [sum(row) for row in B]

        # Order nodes of the smaller graph (A) by degree descending for better pruning
        order = sorted(range(n), key=lambda x: degA[x], reverse=True)

        best_mapping: List[Tuple[int, int]] = []
        used_h = [False] * m  # marks nodes of B already used

        # Recursive backtracking search
        def dfs(idx: int, current: List[Tuple[int, int]]) -> None:
            nonlocal best_mapping

            # Upper bound: remaining nodes that could still be added
            remaining = len(order) - idx
            if len(current) + remaining <= len(best_mapping):
                return  # cannot beat current best

            if idx == len(order):
                # reached end of ordered nodes
                if len(current) > len(best_mapping):
                    best_mapping = current.copy()
                return

            g = order[idx]

            for h in range(m):
                if used_h[h]:
                    continue
                # Check edge consistency with already mapped nodes
                # degree compatibility check removed
                # if degA[g] != degB[h]:
                #     continue
                # Check edge consistency with already mapped nodes
                consistent = True
                for (g2, h2) in current:
                    if A[g][g2] != B[h][h2]:
                        consistent = False
                        break
                if not consistent:
                    continue

                # Choose this mapping
                used_h[h] = True
                current.append((g, h))
                dfs(idx + 1, current)
                current.pop()
                used_h[h] = False

            # Also consider leaving g unmapped
            dfs(idx + 1, current)

        dfs(0, [])

        # If we swapped the graphs, invert the pairs back
        if swapped:
            best_mapping = [(p, g) for (g, p) in best_mapping]

        return best_mapping