from typing import Any

class Solver:
    def solve(self, problem: dict, **kwargs) -> Any:
        """
        Count the number of connected components in an undirected graph.

        Input:
            problem: dict with keys:
                - "edges": iterable of edges; each edge is a 2-tuple (u, v) or a 3-tuple (u, v, attrs)
                - "num_nodes": int-like, number of nodes labeled 0..num_nodes-1

        Output:
            dict with key:
                - "number_connected_components": int
        """
        try:
            # Match reference behavior: raise if "edges" key missing
            edges = problem["edges"]

            # Emulate range(num_nodes) behavior (including exceptions for non-integer types)
            n_raw = problem.get("num_nodes", 0)
            rng = range(n_raw)  # may raise TypeError for invalid n_raw (e.g., float)
            n = len(rng)  # equals n_raw if n_raw >= 0, else 0

            # Disjoint Set Union (Union-Find) with path compression and union by size
            parent = list(range(n))
            size = [1] * n

            # Map arbitrary node labels to DSU indices.
            # Prepopulate with base nodes so that labels equal to ints (e.g., 1.0 == 1) map identically.
            node_index: dict[Any, int] = {i: i for i in rng}

            components = n

            # Local bindings for speed
            parent_list = parent
            size_list = size
            ni_get = node_index.get
            sentinel = object()

            for e in edges:
                # Extract u, v; accept edges of length 2 or 3, reject others to mirror NetworkX behavior
                try:
                    it = iter(e)
                except Exception:
                    return {"number_connected_components": -1}
                try:
                    u = next(it)
                    v = next(it)
                except Exception:
                    return {"number_connected_components": -1}

                extra1 = next(it, sentinel)
                if extra1 is not sentinel:
                    # There is at least a third element; ensure there's not a 4th
                    extra2 = next(it, sentinel)
                    if extra2 is not sentinel:
                        return {"number_connected_components": -1}
                    # exactly 3 elements -> ignore attributes

                # Map or create indices for endpoints
                iu = ni_get(u)
                if iu is None:
                    iu = len(parent_list)
                    node_index[u] = iu
                    parent_list.append(iu)
                    size_list.append(1)
                    components += 1

                iv = ni_get(v)
                if iv is None:
                    iv = len(parent_list)
                    node_index[v] = iv
                    parent_list.append(iv)
                    size_list.append(1)
                    components += 1

                # Find with path halving
                x = iu
                while parent_list[x] != x:
                    parent_list[x] = parent_list[parent_list[x]]
                    x = parent_list[x]
                rx = x

                y = iv
                while parent_list[y] != y:
                    parent_list[y] = parent_list[parent_list[y]]
                    y = parent_list[y]
                ry = y

                if rx != ry:
                    # Union by size
                    if size_list[rx] < size_list[ry]:
                        rx, ry = ry, rx
                    parent_list[ry] = rx
                    size_list[rx] += size_list[ry]
                    components -= 1

            return {"number_connected_components": components}
        except Exception:
            # Use -1 as an unmistakable “solver errored” sentinel
            return {"number_connected_components": -1}