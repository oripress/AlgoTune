from __future__ import annotations

from typing import Any, Dict

try:
    # Compiled Cython extension (built from _edgeexpansion.pyx via setup.py).
    from _edgeexpansion import edge_expansion as _cy_edge_expansion  # type: ignore
except Exception:  # pragma: no cover
    _cy_edge_expansion = None

class Solver:
    __slots__ = ()

    def solve(self, problem: Dict[str, Any], **kwargs) -> Dict[str, float]:
        adj = problem["adjacency_list"]
        nodes_s = problem["nodes_S"]

        # Fast path: Cython implementation
        if _cy_edge_expansion is not None:
            return {"edge_expansion": float(_cy_edge_expansion(adj, nodes_s))}

        # ---- Fallback: optimized pure Python ----
        n = len(adj)
        m = len(nodes_s)
        if n == 0 or m == 0 or m == n:
            return {"edge_expansion": 0.0}

        t = n - m
        denom = m if m < t else t
        if denom == 0:
            return {"edge_expansion": 0.0}

        mask = bytearray(n)
        for u in nodes_s:
            mask[u] = 1
        _mask = mask

        cut = 0
        for u, neigh in enumerate(adj):
            ln = len(neigh)
            if ln == 0:
                continue
            inside = 0
            for v in neigh:
                inside += _mask[v]
            if _mask[u]:
                cut += ln - inside
            else:
                cut += inside

        return {"edge_expansion": cut / denom}