from typing import Any

class Solver:
    def solve(self, problem: list[list[int]], **kwargs) -> list[int]:
        """
        Solves the maximum clique problem using a Bron–Kerbosch algorithm with pivot and branch-and-bound.
        :param problem: Adjacency matrix of the graph (list of list of 0/1).
        :return: List of node indices forming a maximum clique.
        """
        n = len(problem)
        # Build neighbor bitmasks
        nn = [0] * n
        for i in range(n):
            mask = 0
            row = problem[i]
            for j, v in enumerate(row):
                if v:
                    mask |= 1 << j
            nn[i] = mask

        best_size = 0
        best_mask = 0

        # Increase recursion limit
        import sys
        sys.setrecursionlimit(10000)

        def bk(R: int, P: int, X: int):
            nonlocal best_size, best_mask
            # If no candidates and no excluded, R is maximal
            if P == 0 and X == 0:
                size = R.bit_count()
                if size > best_size:
                    best_size = size
                    best_mask = R
                return

            # Branch-and-bound: prune if even taking all P cannot beat best
            if R.bit_count() + P.bit_count() <= best_size:
                return

            # Choose a pivot u from P ∪ X (lowest-index bit)
            UX = P | X
            # lowest bit of UX
            u_bit = UX & -UX
            u = u_bit.bit_length() - 1

            # Candidates are vertices in P not adjacent to pivot u
            cand = P & ~nn[u]
            # Explore candidates
            while cand:
                v_bit = cand & -cand
                v = v_bit.bit_length() - 1
                cand &= cand - 1
                # Recurse with v added to R
                bk(R | v_bit, P & nn[v], X & nn[v])
                # Move v from P to X
                P &= ~v_bit
                X |= v_bit

        # Initial call: R = 0, P = all vertices, X = 0
        all_mask = (1 << n) - 1
        bk(0, all_mask, 0)

        # Extract chosen vertices
        res = []
        mask = best_mask
        while mask:
            v_bit = mask & -mask
            v = v_bit.bit_length() - 1
            res.append(v)
            mask &= mask - 1

        return res