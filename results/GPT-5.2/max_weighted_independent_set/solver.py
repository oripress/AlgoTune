from __future__ import annotations

from typing import Any, List, Tuple

def _lsb_index(x: int) -> int:
    """Index of least-significant set bit; x must be nonzero."""
    return (x & -x).bit_length() - 1

def _iter_bits(mask: int):
    while mask:
        b = mask & -mask
        yield (b.bit_length() - 1)
        mask &= mask - 1

class _MWCliqueSolver:
    """Maximum-weight clique in an undirected graph given as bit-adjacency."""

    __slots__ = ("adj", "w")

    def __init__(self, adj: List[int], w: List[int]):
        self.adj = adj
        self.w = w

    def solve(self) -> List[int]:
        m = len(self.w)
        if m == 0:
            return []
        if m == 1:
            return [0] if self.w[0] > 0 else []

        adj = self.adj
        w = self.w
        allmask = (1 << m) - 1
        allmask = (1 << m) - 1

        # Quick trivial cases
        deg_sum = 0
        complete = True
        for i in range(m):
            d = adj[i].bit_count()
            deg_sum += d
            if d != m - 1:
                complete = False
        if deg_sum == 0:
            # No edges => max clique is best single vertex (or empty if all zero)
            best_v = max(range(m), key=w.__getitem__)
            return [best_v] if w[best_v] > 0 else []
        if complete:
            return [i for i in range(m) if w[i] > 0]  # all vertices form a clique

        # Greedy initial incumbent
        order = sorted(range(m), key=w.__getitem__, reverse=True)
        cand = allmask
        inc: List[int] = []
        inc_w = 0
        for v in order:
            if (cand >> v) & 1:
                inc.append(v)
                inc_w += w[v]
                cand &= adj[v]

        best_w = inc_w
        best = inc[:]  # clique vertices (indices)

        stack = [0] * m

        def color_sort(P: int) -> Tuple[List[int], List[int]]:
            """
            Greedy coloring of the induced subgraph on P.
            Returns:
              verts: vertices ordered by nondecreasing color (color classes packed)
              bounds[i]: upper bound on achievable clique weight within verts[:i+1]
                         using (sum over colors of max weight in that prefix color).
            """
            verts: List[int] = []
            bounds: List[int] = []
            remaining = P
            total = 0

            while remaining:
                avail = remaining
                curmax = 0
                # Build one color class as a maximal independent set (in the graph),
                # hence vertices in the same color are pairwise nonadjacent.
                while avail:
                    lsb = avail & -avail
                    v = lsb.bit_length() - 1
                    # remove v and its neighbors from this color class candidate set
                    avail &= ~((1 << v) | adj[v])
                    remaining &= ~(1 << v)

                    wv = w[v]
                    if wv > curmax:
                        total += wv - curmax
                        curmax = wv
                    verts.append(v)
                    bounds.append(total)

            return verts, bounds

        def expand(P: int, cur_w: int, depth: int) -> None:
            nonlocal best_w, best
            if P == 0:
                if cur_w > best_w:
                    best_w = cur_w
                    best = stack[:depth].copy()
                return

            verts, bounds = color_sort(P)
            P_local = P

            # Explore vertices in reverse (higher colors later => explored first).
            for idx in range(len(verts) - 1, -1, -1):
                if cur_w + bounds[idx] <= best_w:
                    return
                v = verts[idx]
                if ((P_local >> v) & 1) == 0:
                    continue

                stack[depth] = v
                expand(P_local & adj[v], cur_w + w[v], depth + 1)
                P_local &= ~(1 << v)

        expand(allmask, 0, 0)
        return best

class Solver:
    def solve(self, problem: dict, **kwargs) -> Any:
        adj_matrix = problem["adj_matrix"]
        weights = problem["weights"]
        n = len(weights)
        if n == 0:
            return []

        # Build adjacency bitmasks for original graph G.
        adj_masks = [0] * n
        for i, row in enumerate(adj_matrix):
            m = 0
            # row is 0/1 list
            for j, v in enumerate(row):
                if v:
                    m |= 1 << j
            adj_masks[i] = m

        # Decompose MWIS over connected components of G (solutions union).
        unseen = (1 << n) - 1
        solution: List[int] = []

        # Reusable mapping array (reset only for used vertices).
        pos = [-1] * n

        while unseen:
            start = _lsb_index(unseen)
            unseen &= ~(1 << start)

            comp_mask = 1 << start
            frontier = 1 << start

            # BFS via bitmasks
            while frontier:
                u = _lsb_index(frontier)
                frontier &= frontier - 1
                neigh = adj_masks[u] & unseen
                if neigh:
                    unseen &= ~neigh
                    frontier |= neigh
                    comp_mask |= neigh

            comp_vertices = list(_iter_bits(comp_mask))
            # Filter to positive-weight vertices only (never hurts optimality).
            comp_vertices = [v for v in comp_vertices if weights[v] > 0]
            mcomp = len(comp_vertices)
            if mcomp == 0:
                continue
            if mcomp == 1:
                solution.append(comp_vertices[0])
                continue

            # Build induced adjacency on filtered component in compressed indices.
            for i, v in enumerate(comp_vertices):
                pos[v] = i

            comp_all = 0
            for v in comp_vertices:
                comp_all |= 1 << v

            adj_c = [0] * mcomp
            deg_sum = 0
            for i, v in enumerate(comp_vertices):
                neigh_orig = adj_masks[v] & comp_all
                mm = 0
                tmp = neigh_orig
                while tmp:
                    b = tmp & -tmp
                    j = b.bit_length() - 1
                    mm |= 1 << pos[j]
                    tmp &= tmp - 1
                adj_c[i] = mm
                deg_sum += mm.bit_count()

            # Quick special cases on this induced subgraph
            if deg_sum == 0:
                # No edges in G => independent set is all vertices
                solution.extend(comp_vertices)
                for v in comp_vertices:
                    pos[v] = -1
                continue
            if deg_sum == mcomp * (mcomp - 1):
                # Complete graph => independent set size <= 1
                best_v = max(comp_vertices, key=weights.__getitem__)
                solution.append(best_v)
                for v in comp_vertices:
                    pos[v] = -1
                continue

            # Solve MWIS on this component as MW clique in complement graph.
            # Complement adjacency for clique:
            allmask_c = (1 << mcomp) - 1
            comp_adj = [0] * mcomp
            for i in range(mcomp):
                comp_adj[i] = (allmask_c ^ adj_c[i]) & ~(1 << i)

            w_c = [weights[v] for v in comp_vertices]

            # Permute indices: low weight first (so high weight tends to be explored early).
            # Use complement-degree as a mild tie-breaker.
            deg_comp = [comp_adj[i].bit_count() for i in range(mcomp)]
            perm = sorted(range(mcomp), key=lambda i: (w_c[i], deg_comp[i]))
            inv = [0] * mcomp
            for new_i, old_i in enumerate(perm):
                inv[old_i] = new_i

            w_p = [w_c[old_i] for old_i in perm]
            adj_p = [0] * mcomp
            for new_i, old_i in enumerate(perm):
                mask_old = comp_adj[old_i]
                mm = 0
                tmp = mask_old
                while tmp:
                    b = tmp & -tmp
                    j = b.bit_length() - 1
                    mm |= 1 << inv[j]
                    tmp &= tmp - 1
                adj_p[new_i] = mm

            clique_new = _MWCliqueSolver(adj=adj_p, w=w_p).solve()
            # Map back to original vertex ids.
            for new_i in clique_new:
                old_i = perm[new_i]
                solution.append(comp_vertices[old_i])

            for v in comp_vertices:
                pos[v] = -1

        return solution