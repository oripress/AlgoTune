from __future__ import annotations

from typing import Any, Dict, Iterable, Tuple

try:
    # Compiled Cython helper (fast path).
    import uf_cc as _uf_cc  # type: ignore

    _COUNT_CC = _uf_cc.count_cc
except Exception:  # pragma: no cover
    _COUNT_CC = None

class Solver:
    def __init__(self) -> None:
        pass

    def solve(self, problem: Dict[str, Any], **kwargs) -> Any:
        try:
            n = int(problem.get("num_nodes", 0) or 0)
            if n <= 0:
                return {"number_connected_components": 0}
            if n == 1:
                return {"number_connected_components": 1}

            edges: Iterable[Tuple[int, int]] = problem.get("edges") or ()
            if not edges:
                return {"number_connected_components": n}

            if _COUNT_CC is not None:
                return {"number_connected_components": int(_COUNT_CC(n, edges))}

            # Python fallback (DSU with path halving + early exit).
            par = list(range(n))
            sz = [1] * n
            cc = n

            for u, v in edges:
                if u == v:
                    continue

                x = u
                while par[x] != x:
                    par[x] = par[par[x]]
                    x = par[x]
                ru = x

                x = v
                while par[x] != x:
                    par[x] = par[par[x]]
                    x = par[x]
                rv = x

                if ru == rv:
                    continue

                if sz[ru] < sz[rv]:
                    ru, rv = rv, ru
                par[rv] = ru
                sz[ru] += sz[rv]
                cc -= 1

                if cc == 1:
                    break

            return {"number_connected_components": cc}
        except Exception:
            return {"number_connected_components": -1}