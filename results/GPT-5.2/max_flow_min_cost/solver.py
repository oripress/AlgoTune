from __future__ import annotations

from typing import Any, List, Optional, Tuple

import numpy as np

try:
    # OR-Tools provides very fast C++ implementations.
    from ortools.graph.python import min_cost_flow as ort_min_cost_flow

    _HAVE_ORTOOLS = True
except Exception:
    ort_min_cost_flow = None
    _HAVE_ORTOOLS = False

class _Edge:
    __slots__ = ("to", "rev", "cap", "cost", "eid", "sign")

    def __init__(self, to: int, rev: int, cap: float, cost: float, eid: int, sign: int):
        self.to = to
        self.rev = rev
        self.cap = cap
        self.cost = cost
        self.eid = eid  # index into flattened n*n output matrix
        self.sign = sign  # +1 for forward original edge, -1 for reverse of original edge

def _edges_nonzero_filtered(cap: np.ndarray, s: int, t: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Indices (ii, jj) where cap>0, filtered to match validator requirements:
    - no incoming arcs to s
    - no outgoing arcs from t
    - no self-loops
    """
    ii, jj = np.nonzero(cap)
    if ii.size == 0:
        return ii, jj
    mask = (jj != s) & (ii != t) & (ii != jj)
    if mask.all():
        return ii, jj
    return ii[mask], jj[mask]

def _costs_are_int_on_edges(cost_e: np.ndarray) -> bool:
    if np.issubdtype(cost_e.dtype, np.integer):
        return True
    if np.issubdtype(cost_e.dtype, np.floating):
        r = np.rint(cost_e)
        return bool(np.isfinite(cost_e).all() and np.max(np.abs(cost_e - r)) <= 1e-12)
    return False

def _solve_with_ortools_onepass_numpy(
    n: int,
    ii: np.ndarray,
    jj: np.ndarray,
    cap: np.ndarray,
    cost: np.ndarray,
    s: int,
    t: int,
) -> List[List[int]]:
    """
    Single-call min-cost *max*-flow trick:
    Add an extra arc t->s with large negative cost, supplies all 0 (circulation).
    """
    m = int(ii.size)
    if m == 0:
        return [[0] * n for _ in range(n)]

    ub = int(np.sum(cap[s]))
    if ub <= 0:
        return [[0] * n for _ in range(n)]

    cost_e = cost[ii, jj]
    max_c = int(np.max(cost_e)) if cost_e.size else 0
    big = max(1, max_c * n + 1)

    mcf = ort_min_cost_flow.SimpleMinCostFlow()

    starts = ii.tolist()
    ends = jj.tolist()
    caps = cap[ii, jj].astype(np.int64, copy=False).tolist()

    if np.issubdtype(cost.dtype, np.integer):
        unit_costs = cost_e.astype(np.int64, copy=False).tolist()
    else:
        unit_costs = np.rint(cost_e).astype(np.int64, copy=False).tolist()

    # Bulk-add arcs if the binding supports it (much faster than a Python loop).
    add_many = getattr(mcf, "add_arcs_with_capacity_and_unit_cost", None)
    if add_many is not None:
        add_many(starts, ends, caps, unit_costs)
    else:
        add_arc = mcf.add_arc_with_capacity_and_unit_cost
        for u, v, c, w in zip(starts, ends, caps, unit_costs):
            add_arc(u, v, c, w)

    # Extra return arc (not part of output); its index is m (added last).
    mcf.add_arc_with_capacity_and_unit_cost(int(t), int(s), int(ub), int(-big))

    status = mcf.solve()
    if status != mcf.OPTIMAL:
        return [[0] * n for _ in range(n)]

    sol = [[0] * n for _ in range(n)]
    flow = mcf.flow
    for k in range(m):
        f = int(flow(k))
        if f:
            sol[starts[k]][ends[k]] = f
    return sol

def _solve_with_python_mcmf(
    n: int,
    ii: np.ndarray,
    jj: np.ndarray,
    cap: np.ndarray,
    cost: np.ndarray,
    s: int,
    t: int,
) -> List[List[float]]:
    # Successive shortest augmenting path with potentials (supports float costs).
    import heapq
    import math

    g: List[List[_Edge]] = [[] for _ in range(n)]
    flow_flat = [0.0] * (n * n)

    def add_edge(u: int, v: int, c: float, w: float) -> None:
        eid = u * n + v
        fwd = _Edge(v, len(g[v]), float(c), float(w), eid, +1)
        rev = _Edge(u, len(g[u]), 0.0, -float(w), eid, -1)
        g[u].append(fwd)
        g[v].append(rev)

    for u, v in zip(ii.tolist(), jj.tolist()):
        add_edge(int(u), int(v), float(cap[u, v]), float(cost[u, v]))

    pi = [0.0] * n
    inf = float("inf")

    while True:
        dist = [inf] * n
        dist[s] = 0.0
        parent: List[Optional[Tuple[int, int]]] = [None] * n
        h: List[Tuple[float, int]] = [(0.0, s)]

        while h:
            d, u = heapq.heappop(h)
            if d != dist[u]:
                continue
            if u == t:
                break
            pu = pi[u]
            for ei, e in enumerate(g[u]):
                if e.cap <= 0.0:
                    continue
                v = e.to
                nd = d + e.cost + pu - pi[v]
                if nd < dist[v] - 1e-15:
                    dist[v] = nd
                    parent[v] = (u, ei)
                    heapq.heappush(h, (nd, v))

        if parent[t] is None:
            break

        for i in range(n):
            di = dist[i]
            if di < inf:
                pi[i] += di

        addf = inf
        v = t
        while v != s:
            u, ei = parent[v]
            e = g[u][ei]
            if e.cap < addf:
                addf = e.cap
            v = u
        if not math.isfinite(addf) or addf <= 0.0:
            break

        v = t
        while v != s:
            u, ei = parent[v]
            e = g[u][ei]
            e.cap -= addf
            g[v][e.rev].cap += addf
            flow_flat[e.eid] += addf * e.sign
            v = u

    sol = [[0.0] * n for _ in range(n)]
    for u in range(n):
        row = sol[u]
        base = u * n
        for v in range(n):
            f = flow_flat[base + v]
            if f > 0.0:
                row[v] = f
    return sol

class Solver:
    def solve(self, problem: dict, **kwargs) -> Any:
        try:
            s = int(problem["s"])
            t = int(problem["t"])
            cap = np.asarray(problem["capacity"])
            cst = np.asarray(problem["cost"])
            n = int(cap.shape[0])
            if n == 0:
                return []

            ii, jj = _edges_nonzero_filtered(cap, s, t)
            if ii.size == 0:
                return [[0] * n for _ in range(n)]

            if _HAVE_ORTOOLS:
                if np.issubdtype(cst.dtype, np.integer):
                    ok_costs = True
                else:
                    ok_costs = _costs_are_int_on_edges(cst[ii, jj])

                if ok_costs:
                    if not np.issubdtype(cap.dtype, np.integer):
                        cap = np.rint(cap).astype(np.int64, copy=False)
                    else:
                        cap = cap.astype(np.int64, copy=False)
                    return _solve_with_ortools_onepass_numpy(n, ii, jj, cap, cst, s, t)

            cap_f = cap.astype(np.float64, copy=False)
            cost_f = cst.astype(np.float64, copy=False)
            return _solve_with_python_mcmf(n, ii, jj, cap_f, cost_f, s, t)
        except Exception:
            try:
                n = len(problem.get("capacity", []))
            except Exception:
                n = 0
            return [[0 for _ in range(n)] for _ in range(n)]