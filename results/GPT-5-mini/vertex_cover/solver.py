import sys
from typing import Any, List

class Solver:
    def solve(self, problem, **kwargs) -> Any:
        """
        Solve minimum vertex cover by computing a maximum independent set (MIS)
        and returning its complement. The MIS is found as a maximum clique in
        the complement graph using a Tomita-style branch-and-bound with
        greedy coloring for strong pruning.
        """
        n = len(problem)
        if n == 0:
            return []

        # Build bitmask adjacency (no self-loops)
        neighbors = [0] * n
        for i in range(n):
            row = problem[i]
            m = 0
            for j, val in enumerate(row):
                if j != i and val:
                    m |= 1 << j
            neighbors[i] = m

        all_mask = (1 << n) - 1

        # Quick check: if no edges, empty cover
        any_edge = False
        for m in neighbors:
            if m:
                any_edge = True
                break
        if not any_edge:
            return []

        # Greedy maximal independent set to get an initial lower bound (fast)
        def greedy_mis_mask():
            avail = all_mask
            mis = 0
            while avail:
                # pick vertex with minimum degree within available (heuristic)
                best_v = None
                best_deg = n + 1
                tmp = avail
                while tmp:
                    b = tmp & -tmp
                    v = b.bit_length() - 1
                    tmp &= tmp - 1
                    deg = (neighbors[v] & avail).bit_count()
                    if deg < best_deg:
                        best_deg = deg
                        best_v = v
                        if best_deg <= 1:
                            break
                if best_v is None:
                    break
                mis |= 1 << best_v
                # remove chosen vertex and its neighbors
                avail &= ~((1 << best_v) | neighbors[best_v])
            return mis

        # Initial MIS (as mask) and best clique (in complement graph)
        mis_mask = greedy_mis_mask()
        best_clique_mask = mis_mask  # clique in complement graph
        best_clique_size = mis_mask.bit_count()

        # Build complement adjacency bitmasks
        comp_nbr = [0] * n
        for v in range(n):
            comp_nbr[v] = all_mask & ~(neighbors[v] | (1 << v))

        sys.setrecursionlimit(10000)

        # Tomita-style expansion with greedy coloring upper bound
        def tomita_max_clique():
            nonlocal best_clique_size, best_clique_mask

            def expand(R: int, P: int):
                nonlocal best_clique_size, best_clique_mask

                if P == 0:
                    rsize = R.bit_count()
                    if rsize > best_clique_size:
                        best_clique_size = rsize
                        best_clique_mask = R
                    return

                # Build list of vertices in P
                verts = []
                tmp = P
                while tmp:
                    b = tmp & -tmp
                    v = b.bit_length() - 1
                    tmp &= tmp - 1
                    verts.append(v)

                # Order by degree (in P) descending to improve coloring
                verts.sort(key=lambda x: (comp_nbr[x] & P).bit_count(), reverse=True)

                # Greedy coloring to compute an upper bound per vertex
                color_class_masks: List[int] = []
                vertex_color = {}
                for v in verts:
                    assigned = False
                    for ci, cmask in enumerate(color_class_masks):
                        # can place v in this color if it has no neighbors in that class
                        if (comp_nbr[v] & cmask) == 0:
                            color_class_masks[ci] |= 1 << v
                            vertex_color[v] = ci + 1
                            assigned = True
                            break
                    if not assigned:
                        color_class_masks.append(1 << v)
                        vertex_color[v] = len(color_class_masks)

                # Order vertices by color (small to large), process in reverse
                ordered = sorted(verts, key=lambda x: vertex_color[x])
                for v in reversed(ordered):
                    # bounding: if even with all colors we cannot beat best, stop
                    if R.bit_count() + vertex_color[v] <= best_clique_size:
                        return
                    # include v
                    newR = R | (1 << v)
                    newP = P & comp_nbr[v]
                    expand(newR, newP)
                    # remove v from P
                    P &= ~(1 << v)

            expand(0, all_mask)

        try:
            tomita_max_clique()
        except RecursionError:
            # fallback: keep the greedy solution we have
            pass

        # best_clique_mask is an MIS in the original graph (clique in complement)
        mis_mask = best_clique_mask
        cover_mask = all_mask ^ mis_mask
        result: List[int] = [i for i in range(n) if (cover_mask >> i) & 1]
        return result