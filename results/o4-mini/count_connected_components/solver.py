from typing import Any

class Solver:
    def solve(self, problem: dict[str, Any], **kwargs) -> Any:
        n = problem.get("num_nodes", 0)
        edges = problem.get("edges", [])
        if n <= 0:
            return {"number_connected_components": 0}
        parent = list(range(n))
        rank = [0] * n
        comp = n
        for u, v in edges:
            # find root of u with path compression
            x = u
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            ru = x
            # find root of v with path compression
            x = v
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            rv = x
            if ru != rv:
                if rank[ru] < rank[rv]:
                    parent[ru] = rv
                else:
                    parent[rv] = ru
                    if rank[ru] == rank[rv]:
                        rank[ru] += 1
                comp -= 1
                if comp == 1:
                    break
        return {"number_connected_components": comp}