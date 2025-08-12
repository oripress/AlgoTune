from typing import Any, List, Tuple

class Solver:
    def solve(self, problem: dict[str, Any]) -> dict[str, List[List[float]]]:
        """
        Compute the Minimum Spanning Tree (MST) of an undirected weighted graph
        using Kruskal's algorithm with a Union-Find data structure.
        The result edges are returned as [u, v, weight] with u < v,
        sorted by (u, v) for deterministic output matching the reference.
        """
        num_nodes: int = problem["num_nodes"]
        edges: List[Tuple[int, int, float]] = problem["edges"]

        # Union-Find (Disjoint Set Union) implementation
        parent = list(range(num_nodes))
        rank = [0] * num_nodes

        def find(x: int) -> int:
            # Path compression
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        def union(x: int, y: int) -> bool:
            rx, ry = find(x), find(y)
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

        # Stable sort edges by weight (preserves input order for equal weights)
        sorted_edges = sorted(edges, key=lambda e: e[2])

        mst_edges: List[List[float]] = []
        for u, v, w in sorted_edges:
            if union(u, v):
                # Ensure consistent ordering of node indices
                if u > v:
                    u, v = v, u
                mst_edges.append([float(u), float(v), float(w)])
                if len(mst_edges) == num_nodes - 1:
                    break

        # Final sorting by (u, v) as required by the specification
        mst_edges.sort(key=lambda e: (e[0], e[1]))
        return {"mst_edges": mst_edges}