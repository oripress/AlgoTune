from typing import Any, Dict, List, Tuple

class Solver:
    def solve(self, problem: Dict[str, Any], **kwargs) -> Dict[str, List[List[float]]]:
        """
        Compute the Minimum Spanning Tree (or forest) of an undirected weighted graph.

        The implementation mimics NetworkX's behavior for:
        - Single edge per undirected pair (last weight wins for duplicates)
        - Tie-breaking in Kruskal via stable sort by weight with original Graph.edges order,
          which we reconstruct by simulating NetworkX's undirected Graph edge iteration:
            * nodes are in insertion order 0..num_nodes-1
            * for each node u in that order, neighbors are in the order first seen
            * undirected edges are yielded once with (u, v) where v >= u (self-loops included)

        Returns edges with endpoints ordered (u < v) and sorted by (u, v).
        """
        num_nodes = int(problem.get("num_nodes", 0))
        edges_in: List[List[float]] = problem.get("edges", [])

        # Quick exits
        if num_nodes <= 1 or not edges_in:
            return {"mst_edges": []}

        # Track:
        # - first_seen: whether an undirected pair (a, b) has been seen (a <= b)
        # - final_weight: the last weight seen for undirected pair (a, b) (a <= b)
        # - adj: neighbor insertion order lists to reconstruct Graph.edges order
        first_seen: Dict[Tuple[int, int], bool] = {}
        final_weight: Dict[Tuple[int, int], float] = {}
        adj: List[List[int]] = [[] for _ in range(num_nodes)]

        for u_raw, v_raw, w_raw in edges_in:
            u = int(u_raw)
            v = int(v_raw)
            # Normalize pair as undirected key (a <= b)
            if u <= v:
                a, b = u, v
            else:
                a, b = v, u

            # Record first time seeing this undirected edge to mimic neighbor insertion order
            key = (a, b)
            if key not in first_seen:
                first_seen[key] = True
                # For undirected Graph adjacency, insert neighbor once to each side
                adj[a].append(b)
                if a != b:
                    adj[b].append(a)

            # In networkx.Graph, adding the same edge updates attributes -> last weight wins
            # Ensure float type to match expected formatting
            final_weight[key] = float(w_raw)

        # Reconstruct the order in which networkx.Graph.edges() would yield edges:
        # Iterate nodes in insertion order 0..num_nodes-1, then neighbors in insertion order.
        # For undirected, yield only when v >= u (self-loops included).
        order_idx: Dict[Tuple[int, int], int] = {}
        order_counter = 0
        for u in range(num_nodes):
            for v in adj[u]:
                if v < u:
                    continue
                order_idx[(u, v)] = order_counter
                order_counter += 1

        # Build list of edges with (weight, order, a, b) for Kruskal with networkx tie-breaking
        # Only include pairs that exist (present in order_idx).
        sortable_edges: List[Tuple[float, int, int, int]] = []
        for (a, b), idx in order_idx.items():
            # Edge must exist in final_weight; if not, it was never provided (shouldn't happen)
            w = final_weight.get((a, b))
            if w is None:
                # If an edge was "first seen" but later removed (not possible here), skip
                continue
            sortable_edges.append((w, idx, a, b))

        # Sort by weight, then by reconstructed Graph.edges order (stable tie-breaking)
        sortable_edges.sort(key=lambda t: (t[0], t[1]))

        # Disjoint Set Union (Union-Find) for Kruskal
        parent = list(range(num_nodes))
        rank = [0] * num_nodes

        def find(x: int) -> int:
            # Path compression
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        def union(x: int, y: int) -> bool:
            rx = find(x)
            ry = find(y)
            if rx == ry:
                return False
            # Union by rank
            if rank[rx] < rank[ry]:
                parent[rx] = ry
            elif rank[rx] > rank[ry]:
                parent[ry] = rx
            else:
                parent[ry] = rx
                rank[rx] += 1
            return True

        # Kruskal's algorithm
        mst_edges: List[List[float]] = []
        for w, _, a, b in sortable_edges:
            if a == b:
                # Self-loop never connects components
                continue
            if union(a, b):
                # Ensure u < v for consistency
                u, v = (a, b) if a < b else (b, a)
                mst_edges.append([u, v, w])

        # Sort final MST edges by (u, v) as required
        mst_edges.sort(key=lambda x: (x[0], x[1]))
        return {"mst_edges": mst_edges}