from typing import Any

class Solver:
    def solve(self, problem: dict, **kwargs) -> Any:
        """
        A maximally optimized pure Python implementation of Kruskal's algorithm.
        This version builds upon the fastest previous solution by inlining the
        Disjoint Set Union (DSU) logic directly into the main loop. By
        eliminating the function call overhead for `find` and `union` inside
        the hot loop, this approach aims to reduce constant factor overheads
        to an absolute minimum.
        """
        num_nodes = problem["num_nodes"]
        edges = problem["edges"]

        if num_nodes <= 1 or not edges:
            return {"mst_edges": []}

        # 1. In-place sort by weight. This is the O(E log E) bottleneck, but
        # Python's Timsort is highly optimized C code.
        edges.sort(key=lambda x: x[2])

        # 2. Initialize DSU data structures.
        parent = list(range(num_nodes))
        size = [1] * num_nodes
        
        mst_edges = []
        edge_count = 0
        
        # 3. Main loop with inlined DSU operations to avoid function call overhead.
        for u, v, weight in edges:
            # --- Inlined find(u) ---
            root_u = u
            while parent[root_u] != root_u:
                root_u = parent[root_u]
            # Path compression for u's path
            path_node = u
            while path_node != root_u:
                next_node = parent[path_node]
                parent[path_node] = root_u
                path_node = next_node

            # --- Inlined find(v) ---
            root_v = v
            while parent[root_v] != root_v:
                root_v = parent[root_v]
            # Path compression for v's path
            path_node = v
            while path_node != root_v:
                next_node = parent[path_node]
                parent[path_node] = root_v
                path_node = next_node

            # --- Inlined union logic ---
            if root_u != root_v:
                # Union by size
                if size[root_u] < size[root_v]:
                    root_u, root_v = root_v, root_u
                parent[root_v] = root_u
                size[root_u] += size[root_v]
                
                # Edge is part of MST
                if u > v:
                    u, v = v, u
                mst_edges.append([u, v, weight])
                edge_count += 1
                if edge_count == num_nodes - 1:
                    break
        
        # 4. Final sort for canonical output.
        mst_edges.sort()
        return {"mst_edges": mst_edges}