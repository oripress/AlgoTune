from typing import Any

try:
    from ccsolver import count_components as _count_components_c
except Exception:
    _count_components_c = None

if _count_components_c is not None:

    class Solver:
        __slots__ = ()

        def solve(self, problem, **kwargs) -> Any:
            return {
                "number_connected_components": _count_components_c(
                    problem["num_nodes"], problem["edges"]
                )
            }

else:

    class Solver:
        __slots__ = ()

        def solve(self, problem, **kwargs) -> Any:
            try:
                n = problem.get("num_nodes", 0)
                edges = problem["edges"]

                if n <= 1:
                    return {"number_connected_components": n}
                if not edges:
                    return {"number_connected_components": n}

                parent = [-1] * n
                components = n

                try:
                    for u, v in edges:
                        x = u
                        while True:
                            px = parent[x]
                            if px < 0:
                                break
                            gx = parent[px]
                            if gx >= 0:
                                parent[x] = gx
                            x = px

                        y = v
                        while True:
                            py = parent[y]
                            if py < 0:
                                break
                            gy = parent[py]
                            if gy >= 0:
                                parent[y] = gy
                            y = py

                        if x != y:
                            if parent[x] > parent[y]:
                                x, y = y, x
                            parent[x] += parent[y]
                            parent[y] = x
                            components -= 1
                            if components == 1:
                                break
                except Exception:
                    return self._solve_fallback(n, edges)

                return {"number_connected_components": components}
            except Exception:
                return {"number_connected_components": -1}

        @staticmethod
        def _solve_fallback(n: int, edges) -> dict[str, int]:
            parent = {}
            size = {}

            for i in range(n):
                parent[i] = i
                size[i] = 1

            components = n

            for u, v in edges:
                if u not in parent:
                    parent[u] = u
                    size[u] = 1
                    components += 1
                if v not in parent:
                    parent[v] = v
                    size[v] = 1
                    components += 1

                pu = u
                while parent[pu] != pu:
                    parent[pu] = parent[parent[pu]]
                    pu = parent[pu]

                pv = v
                while parent[pv] != pv:
                    parent[pv] = parent[parent[pv]]
                    pv = parent[pv]

                if pu != pv:
                    if size[pu] < size[pv]:
                        pu, pv = pv, pu
                    parent[pv] = pu
                    size[pu] += size[pv]
                    components -= 1

            return {"number_connected_components": components}