import sys
from typing import Any, List

sys.setrecursionlimit(10000)

class Solver:
    def solve(self, problem: List[List[int]], **kwargs) -> Any:
        """
        Exact maximum clique solver using bitset operations and a Tomita-style
        branch-and-bound with greedy coloring for pruning.

        :param problem: adjacency matrix (list of lists) with 0/1 entries (symmetric)
        :return: list of original vertex indices forming a maximum clique
        """
        if problem is None:
            return []
        n = len(problem)
        if n == 0:
            return []
        if n == 1:
            return [0]

        # Degrees (ignore diagonal)
        deg = [0] * n
        for i, row in enumerate(problem):
            s = 0
            for j, val in enumerate(row):
                if j != i and val:
                    s += 1
            deg[i] = s

        # Simple ordering by degree descending (good heuristic)
        order = sorted(range(n), key=lambda x: deg[x], reverse=True)

        old_to_new = [0] * n
        new_to_old = [0] * n
        for new_idx, old_idx in enumerate(order):
            old_to_new[old_idx] = new_idx
            new_to_old[new_idx] = old_idx

        # Build adjacency bitmasks in new indexing
        adj: List[int] = [0] * n
        for new_i in range(n):
            old_i = new_to_old[new_i]
            row = problem[old_i]
            mask = 0
            for old_j, val in enumerate(row):
                if val:
                    mask |= (1 << old_to_new[old_j])
            mask &= ~(1 << new_i)
            adj[new_i] = mask

        # Helper: convert bitmask to list of vertices (ascending)
        def mask_to_list(mask: int) -> List[int]:
            res: List[int] = []
            while mask:
                lsb = mask & -mask
                v = lsb.bit_length() - 1
                res.append(v)
                mask ^= lsb
            return res

        # Initial greedy to obtain lower bound
        best_mask = 0
        best_size = 0

        # Greedy starting from each of first min(n, 60) vertices
        limit = min(n, 60)
        for start in range(limit):
            v = start
            cur_mask = 1 << v
            cand = adj[v]
            while cand:
                # pick vertex in cand with max neighbors inside cand
                m = cand
                best_u = -1
                best_score = -1
                while m:
                    lsb = m & -m
                    u = lsb.bit_length() - 1
                    m ^= lsb
                    score = (adj[u] & cand).bit_count()
                    if score > best_score:
                        best_score = score
                        best_u = u
                if best_u == -1:
                    break
                cur_mask |= (1 << best_u)
                cand &= adj[best_u]
            cur_size = cur_mask.bit_count()
            if cur_size > best_size:
                best_size = cur_size
                best_mask = cur_mask

        # Another simple greedy: scan vertices and add if compatible
        cur_mask = 0
        for v in range(n):
            ok = True
            m = cur_mask
            while m:
                lsb = m & -m
                u = lsb.bit_length() - 1
                if not ((adj[v] >> u) & 1):
                    ok = False
                    break
                m ^= lsb
            if ok:
                cur_mask |= (1 << v)
        cur_size = cur_mask.bit_count()
        if cur_size > best_size:
            best_size = cur_size
            best_mask = cur_mask

        adj_local = adj  # alias

        # Recursive branch and bound with coloring
        def expand(R_mask: int, R_size: int, P_mask: int):
            nonlocal best_size, best_mask, adj_local
            if P_mask == 0:
                if R_size > best_size:
                    best_size = R_size
                    best_mask = R_mask
                return

            # List vertices in P
            vertices: List[int] = []
            pm = P_mask
            while pm:
                lsb = pm & -pm
                v = lsb.bit_length() - 1
                vertices.append(v)
                pm ^= lsb

            # Order vertices by degree inside P (descending)
            vertices.sort(key=lambda x: (adj_local[x] & P_mask).bit_count(), reverse=True)

            # Greedy coloring: assign colors so same-color vertices are pairwise non-adjacent
            color_of = {}
            color_masks: List[int] = []
            for v in vertices:
                placed = False
                for ci, cm in enumerate(color_masks):
                    if (adj_local[v] & cm) == 0:
                        color_of[v] = ci + 1
                        color_masks[ci] |= (1 << v)
                        placed = True
                        break
                if not placed:
                    color_of[v] = len(color_masks) + 1
                    color_masks.append(1 << v)

            # Process vertices in increasing color order, but explore from last to first
            ordered = sorted(vertices, key=lambda x: color_of[x])
            P = P_mask
            for v in reversed(ordered):
                # pruning: optimistic bound
                if R_size + color_of[v] <= best_size:
                    return
                P &= ~(1 << v)
                newR = R_mask | (1 << v)
                newP = P_mask & adj_local[v]
                if newP:
                    expand(newR, R_size + 1, newP)
                else:
                    if R_size + 1 > best_size:
                        best_size = R_size + 1
                        best_mask = newR

        full_mask = (1 << n) - 1
        expand(0, 0, full_mask)

        # Map back to original indices
        vertices = mask_to_list(best_mask)
        result = [new_to_old[v] for v in vertices]
        result.sort()
        return result