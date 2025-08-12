from typing import Any, Dict, List

class Solver:
    def solve(self, problem: Dict[str, Any], **kwargs) -> Dict[str, List[List[float]]]:
        """
        Compute the Minimum Spanning Forest using Kruskal's algorithm.
        Preserves adjacency-based insertion/order behavior so results match the reference.
        Returns {"mst_edges": [[u, v, weight], ...]} sorted by (u, v).
        """
        num_nodes = int(problem.get("num_nodes", 0) or 0)
        raw_edges = problem.get("edges", []) or []

        if num_nodes <= 0 or not raw_edges:
            return {"mst_edges": []}

        # Build adjacency mapping; later duplicate edges overwrite earlier ones
        adj: List[Dict[int, float]] = [dict() for _ in range(num_nodes)]
        for e in raw_edges:
            if not isinstance(e, (list, tuple)) or len(e) < 3:
                continue
            try:
                u = int(e[0])
                v = int(e[1])
                w = float(e[2])
            except Exception:
                continue
            if not (0 <= u < num_nodes and 0 <= v < num_nodes):
                continue
            if u == v:
                continue
            adj[u][v] = w
            adj[v][u] = w

        # Collect unique undirected edges (u <= v) in node order and neighbor insertion order
        edges: List[tuple] = []
        for u in range(num_nodes):
            for v, w in adj[u].items():
                if u <= v:
                    edges.append((u, v, w))

        # Stable sort by weight so ties respect insertion order
        edges.sort(key=lambda x: x[2])

        # Union-Find (inlined find with path halving + union by rank)
        parent = list(range(num_nodes))
        rank = [0] * num_nodes
        p = parent
        r = rank

        mst: List[List[float]] = []
        need = max(0, num_nodes - 1)
        taken = 0

        for u, v, w in edges:
            # find root u (path halving)
            x = u
            while p[x] != x:
                p[x] = p[p[x]]
                x = p[x]
            ru = x
            # find root v (path halving)
            y = v
            while p[y] != y:
                p[y] = p[p[y]]
                y = p[y]
            rv = y

            if ru == rv:
                continue

            # union by rank
            if r[ru] < r[rv]:
                p[ru] = rv
            elif r[ru] > r[rv]:
                p[rv] = ru
            else:
                p[rv] = ru
                r[ru] += 1

            mst.append([u, v, w])
            taken += 1
            if taken == need:
                break

        mst.sort(key=lambda e: (e[0], e[1]))
        return {"mst_edges": mst}