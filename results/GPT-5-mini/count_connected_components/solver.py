from typing import Any, Dict

class Solver:
    def solve(self, problem: Dict[str, Any], **kwargs) -> Dict[str, int]:
        """
        Count connected components in an undirected graph.

        Nodes considered are the union of range(num_nodes) and any nodes appearing in edges.
        Uses an optimized union-find (disjoint-set) with negative-size roots and an
        on-the-fly mapping for nodes not covered by 0..num_nodes-1 to avoid building
        large intermediate sets.
        """
        data = problem or {}
        edges = data.get("edges", []) or []
        # Normalize edges to a list if it's another iterable
        if not isinstance(edges, list):
            try:
                edges = list(edges)
            except Exception:
                edges = []

        # Parse num_nodes safely
        n = data.get("num_nodes", 0)
        try:
            n = int(n)
        except Exception:
            n = 0
        if n < 0:
            n = 0

        # Quick path: no edges -> all nodes are isolated
        if not edges:
            return {"number_connected_components": n}

        # Fast path: if all endpoints are ints in [0, n-1], use a simple array-based DSU
        can_use_array = True
        for e in edges:
            try:
                u, v = e
            except Exception:
                can_use_array = False
                break
            if not isinstance(u, int) or not isinstance(v, int) or u < 0 or v < 0 or u >= n or v >= n:
                can_use_array = False
                break

        # Helper: negative-size parent array DSU with path-halving
        if can_use_array:
            parent = [-1] * n  # negative size for roots
            comp = n
            p = parent  # local alias
            for u, v in edges:
                if u == v:
                    continue
                # find root u
                x = u
                while p[x] >= 0:
                    if p[p[x]] >= 0:
                        p[x] = p[p[x]]
                    x = p[x]
                ru = x
                # find root v
                x = v
                while p[x] >= 0:
                    if p[p[x]] >= 0:
                        p[x] = p[p[x]]
                    x = p[x]
                rv = x
                if ru == rv:
                    continue
                # union by size (more negative => larger)
                if p[ru] < p[rv]:
                    p[ru] += p[rv]
                    p[rv] = ru
                else:
                    p[rv] += p[ru]
                    p[ru] = rv
                comp -= 1
            return {"number_connected_components": comp}

        # General case: start with nodes 0..n-1, and add new nodes on demand
        parent = [-1] * n if n > 0 else []
        p = parent
        comp = n
        idx_map = {}
        m = n  # current number of distinct nodes (size of parent)

        for e in edges:
            try:
                u, v = e
            except Exception:
                continue

            # get index for u
            if isinstance(u, int) and 0 <= u < n:
                iu = u
            else:
                try:
                    iu = idx_map[u]
                except Exception:
                    # either KeyError (not present) or TypeError (unhashable)
                    try:
                        # try to use the object as a key first (if hashable)
                        if u not in idx_map:
                            idx_map[u] = m
                            iu = m
                            m += 1
                            p.append(-1)
                            comp += 1
                        else:
                            iu = idx_map[u]
                    except Exception:
                        # fallback for unhashable objects: use repr()
                        key = repr(u)
                        if key in idx_map:
                            iu = idx_map[key]
                        else:
                            idx_map[key] = m
                            iu = m
                            m += 1
                            p.append(-1)
                            comp += 1

            # get index for v
            if isinstance(v, int) and 0 <= v < n:
                iv = v
            else:
                try:
                    iv = idx_map[v]
                except Exception:
                    try:
                        if v not in idx_map:
                            idx_map[v] = m
                            iv = m
                            m += 1
                            p.append(-1)
                            comp += 1
                        else:
                            iv = idx_map[v]
                    except Exception:
                        key = repr(v)
                        if key in idx_map:
                            iv = idx_map[key]
                        else:
                            idx_map[key] = m
                            iv = m
                            m += 1
                            p.append(-1)
                            comp += 1

            # union iu and iv
            if iu == iv:
                continue
            # find root iu
            x = iu
            while p[x] >= 0:
                if p[p[x]] >= 0:
                    p[x] = p[p[x]]
                x = p[x]
            ru = x
            # find root iv
            x = iv
            while p[x] >= 0:
                if p[p[x]] >= 0:
                    p[x] = p[p[x]]
                x = p[x]
            rv = x
            if ru == rv:
                continue
            if p[ru] < p[rv]:
                p[ru] += p[rv]
                p[rv] = ru
            else:
                p[rv] += p[ru]
                p[ru] = rv
            comp -= 1

        return {"number_connected_components": comp}