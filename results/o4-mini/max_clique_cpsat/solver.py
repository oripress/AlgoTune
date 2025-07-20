class Solver:
    def solve(self, problem, **kwargs):
        n = len(problem)
        if n == 0:
            return []
        import sys
        sys.setrecursionlimit(10000)
        # Build adjacency bitmasks
        neighbors = [0] * n
        for i, row in enumerate(problem):
            mask = 0
            for j, val in enumerate(row):
                if val:
                    mask |= 1 << j
            neighbors[i] = mask
        best_clique = 0
        best_set = 0
        # Recursive Bron–Kerbosch with pivoting and pruning
        def bk(R, P, X):
            nonlocal best_clique, best_set
            if P == 0 and X == 0:
                size_r = R.bit_count()
                if size_r > best_clique:
                    best_clique = size_r
                    best_set = R
                return
            # Prune if cannot beat current best
            if R.bit_count() + P.bit_count() <= best_clique:
                return

            # Coloring bound to further prune (greedy coloring)
            col_bound = 0
            uncolored = P
            while uncolored:
                col_bound += 1
                tmp_mask = uncolored
                while tmp_mask:
                    v_bit = tmp_mask & -tmp_mask
                    tmp_mask ^= v_bit
                    v = v_bit.bit_length() - 1
                    uncolored ^= v_bit
                    tmp_mask &= ~neighbors[v]
            if R.bit_count() + col_bound <= best_clique:
                return

            # Pivot choice: choose pivot u in P∪X with maximal connections to P
            union_px = P | X
            max_deg = -1
            u = -1
            tmp_u = union_px
            while tmp_u:
                ub = tmp_u & -tmp_u
                tmp_u ^= ub
                idx = ub.bit_length() - 1
                deg = (P & neighbors[idx]).bit_count()
                if deg > max_deg:
                    max_deg = deg
                    u = idx
            # Vertices in P not adjacent to pivot
            candidates = P & ~neighbors[u]
            while candidates:
                v_bit = candidates & -candidates
                candidates ^= v_bit
                v = v_bit.bit_length() - 1
                bk(R | v_bit, P & neighbors[v], X & neighbors[v])
                P ^= v_bit
                X |= v_bit
        # Start with R=∅, P=all vertices, X=∅
        P = (1 << n) - 1
        bk(0, P, 0)
        # Extract resulting clique
        result = []
        mask = best_set
        while mask:
            v_bit = mask & -mask
            v = v_bit.bit_length() - 1
            result.append(v)
            mask ^= v_bit
        result.sort()
        return result