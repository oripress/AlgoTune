from typing import Any, Dict, List

class Solver:
    def solve(self, problem: Dict[str, Any], **kwargs) -> Dict[str, List[List[float]]]:
        """
        Fast MST solver replicating NetworkX's minimum_spanning_edges behavior:
        - Undirected simple graph semantics (last write wins for duplicate edges)
        - Tie-breaking among equal weights matches NetworkX stable sort order,
          which is the order of G.edges(): nodes in insertion order (0..n-1),
          neighbors in insertion order from the first time an edge is added.

        Returns edges with u < v and sorted by (u, v).
        """
        num_nodes = int(problem.get("num_nodes", 0))
        edges_input = problem.get("edges", [])
        if num_nodes <= 1 or not edges_input:
            return {"mst_edges": []}

        n = num_nodes

        # Per-anchor adjacency with neighbor insertion order and last-write-wins weights.
        neighbors: List[List[int]] = [[] for _ in range(n)]
        weights: List[List[float]] = [[] for _ in range(n)]
        pos_map: List[Dict[int, int]] = [dict() for _ in range(n)]
        neighbors_l = neighbors
        weights_l = weights
        pos_map_l = pos_map

        for u, v, w in edges_input:
            # normalize undirected edge and skip self-loops early
            if u <= v:
                a, b = u, v
            else:
                a, b = v, u
            if a == b:
                continue

            d = pos_map_l[a]
            pos = d.get(b)
            if pos is None:
                pos = len(neighbors_l[a])
                d[b] = pos
                neighbors_l[a].append(b)
                weights_l[a].append(w)
            else:
                weights_l[a][pos] = w  # last write wins

        # Build arrays of edges in G.edges() order (anchors 0..n-1, neighbors first-seen)
        a_list: List[int] = []
        b_list: List[int] = []
        w_list: List[float] = []
        al_extend = a_list.extend
        bl_extend = b_list.extend
        wl_extend = w_list.extend

        for a in range(n):
            nb = neighbors_l[a]
            if not nb:
                continue
            wt = weights_l[a]
            ln = len(nb)
            bl_extend(nb)
            wl_extend(wt)
            al_extend([a] * ln)

        m = len(w_list)
        if m == 0:
            return {"mst_edges": []}

        # Stable ordering of indices by weight only
        use_np = m >= 10_000
        if use_np:
            try:
                import numpy as np  # lazy import
                order = np.argsort(np.asarray(w_list, dtype=float), kind="mergesort")
            except Exception:
                order = None
        else:
            order = None

        if order is None:
            order = list(range(m))
            wl_get = w_list.__getitem__
            order.sort(key=wl_get)

        # Union-Find (single array: negative size for roots, else parent index)
        uf = [-1] * n
        uf_l = uf

        components = n
        mst_edges: List[List[float]] = []
        me = mst_edges.append

        A = a_list
        B = b_list
        W = w_list

        for idx in order:
            idx = int(idx)  # works for both Python int and NumPy scalar
            a = A[idx]
            b = B[idx]

            # find(a) with path halving
            x = a
            while uf_l[x] >= 0:
                px = uf_l[x]
                ppx = uf_l[px]
                if ppx >= 0:
                    uf_l[x] = ppx
                x = px
            rx = x

            # find(b)
            y = b
            while uf_l[y] >= 0:
                py = uf_l[y]
                ppy = uf_l[py]
                if ppy >= 0:
                    uf_l[y] = ppy
                y = py
            ry = y

            if rx != ry:
                # union by size (sizes are negative)
                if uf_l[rx] > uf_l[ry]:
                    rx, ry = ry, rx
                uf_l[rx] += uf_l[ry]
                uf_l[ry] = rx
                me([a, b, W[idx]])
                components -= 1
                if components == 1:
                    break

        # Already ensured a <= b; sort by (u, v) as required
        mst_edges.sort()
        return {"mst_edges": mst_edges}