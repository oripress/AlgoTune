from typing import List, Tuple

class Solver:
    def solve(self, problem: dict, **kwargs) -> List[Tuple[int, int]]:
        """
        Maximum common induced subgraph via maximum clique on the modular product graph.
        Vertices of the product graph are pairs (i, p) with i in G (A) and p in H (B).
        Two vertices (i,p) and (j,q) are adjacent iff i != j, p != q and A[i][j] == B[p][q].
        We find a maximum clique using a bitset-based Tomita-style branch-and-bound
        with greedy coloring for an upper bound.
        """
        A = problem.get("A", [])
        B = problem.get("B", [])
        n = len(A)
        m = len(B)
        if n == 0 or m == 0:
            return []

        N = n * m
        # adjacency bitsets for product graph
        adj: List[int] = [0] * N

        Amat = A
        Bmat = B
        m_local = m

        # Build adjacency: consider i<j and p<q and add both cross edges when A[i][j] == B[p][q]
        for i in range(n - 1):
            Ai = Amat[i]
            for j in range(i + 1, n):
                aij = Ai[j]
                base_j = j * m_local
                base_i = i * m_local
                for p in range(m_local - 1):
                    Bp = Bmat[p]
                    u = base_i + p
                    for q in range(p + 1, m_local):
                        if aij == Bp[q]:
                            v = base_j + q
                            adj[u] |= (1 << v)
                            adj[v] |= (1 << u)
                            u2 = base_i + q
                            v2 = base_j + p
                            adj[u2] |= (1 << v2)
                            adj[v2] |= (1 << u2)

        best_clique_mask = 0
        best_size = 0

        # Helper: iterate set bits of integer
        def iter_bits(x: int):
            while x:
                lsb = x & -x
                idx = lsb.bit_length() - 1
                yield idx
                x ^= lsb

        # Main recursive expansion (R: current clique mask, P: candidate mask)
        def expand(R: int, P: int):
            nonlocal best_clique_mask, best_size

            if P == 0:
                rsize = R.bit_count()
                if rsize > best_size:
                    best_size = rsize
                    best_clique_mask = R
                return

            rsize = R.bit_count()
            pcount = P.bit_count()
            # bound: even if take all P, cannot beat best
            if rsize + pcount <= best_size:
                return

            # build list of vertices in P
            vertices = [v for v in iter_bits(P)]

            # heuristic ordering by degree within P (descending)
            deg_in_p = [(adj[v] & P).bit_count() for v in vertices]
            order = list(range(len(vertices)))
            order.sort(key=lambda i: deg_in_p[i], reverse=True)
            vertices = [vertices[i] for i in order]

            # greedy coloring to get upper bounds (colors start at 1)
            pos = {v: i for i, v in enumerate(vertices)}
            colors = [0] * len(vertices)
            uncolored = vertices[:]
            color = 0
            while uncolored:
                color += 1
                forbidden = 0
                next_uncolored = []
                for v in uncolored:
                    # if v is not forbidden for this color class
                    if ((forbidden >> v) & 1) == 0:
                        colors[pos[v]] = color
                        forbidden |= (adj[v] | (1 << v))
                    else:
                        next_uncolored.append(v)
                uncolored = next_uncolored

            # process vertices in reverse color order (largest color first)
            idxs = list(range(len(vertices)))
            idxs.sort(key=lambda i: colors[i])  # ascending colors
            # iterate from largest color to smallest
            for ii in range(len(idxs) - 1, -1, -1):
                i = idxs[ii]
                v = vertices[i]
                c = colors[i]
                # bound check: if even with this color we can't beat best, prune entirely
                if rsize + c <= best_size:
                    return
                # branch: include v
                newR = R | (1 << v)
                newP = P & adj[v]
                expand(newR, newP)
                # exclude v from P (so subsequent iterations don't reconsider it)
                P &= ~(1 << v)

        allP = (1 << N) - 1 if N > 0 else 0
        expand(0, allP)

        # Convert best_clique_mask to list of (i,p) pairs
        res: List[Tuple[int, int]] = []
        mask = best_clique_mask
        while mask:
            lsb = mask & -mask
            v = lsb.bit_length() - 1
            i = v // m_local
            p = v % m_local
            res.append((i, p))
            mask ^= lsb

        return res