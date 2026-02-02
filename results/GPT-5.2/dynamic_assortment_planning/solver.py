from __future__ import annotations

from heapq import heappop, heappush
from typing import Any, List, Tuple

import numpy as np
from scipy.optimize import linear_sum_assignment

class _Edge:
    __slots__ = ("to", "rev", "cap", "cost")

    def __init__(self, to: int, rev: int, cap: int, cost: float) -> None:
        self.to = to
        self.rev = rev
        self.cap = cap
        self.cost = cost

class _MinCostFlow:
    """
    Successive shortest augmenting path with Johnson potentials.
    Costs are floats; capacities are ints.

    Fallback exact solver when the assignment-matrix expansion is too large.
    """

    __slots__ = ("n", "g")

    def __init__(self, n: int) -> None:
        self.n = n
        self.g: List[List[_Edge]] = [[] for _ in range(n)]

    def add_edge(self, fr: int, to: int, cap: int, cost: float) -> None:
        fwd = _Edge(to, len(self.g[to]), cap, cost)
        rev = _Edge(fr, len(self.g[fr]), 0, -cost)
        self.g[fr].append(fwd)
        self.g[to].append(rev)

    def min_cost_flow(self, s: int, t: int, maxf: int) -> Tuple[int, float]:
        n = self.n
        g = self.g

        pot = [0.0] * n
        prevv = [-1] * n
        preve = [-1] * n
        inf = float("inf")

        flow = 0
        cost = 0.0

        while flow < maxf:
            dist = [inf] * n
            dist[s] = 0.0
            hq: List[Tuple[float, int]] = [(0.0, s)]

            while hq:
                d, v = heappop(hq)
                if d != dist[v]:
                    continue
                gv = g[v]
                pv = pot[v]
                for ei in range(len(gv)):
                    e = gv[ei]
                    if e.cap <= 0:
                        continue
                    nd = d + e.cost + pv - pot[e.to]
                    if nd < dist[e.to] - 1e-12:
                        dist[e.to] = nd
                        prevv[e.to] = v
                        preve[e.to] = ei
                        heappush(hq, (nd, e.to))

            if dist[t] == inf:
                break

            for v in range(n):
                dv = dist[v]
                if dv != inf:
                    pot[v] += dv

            addf = 1
            v = t
            while v != s:
                pv = prevv[v]
                ei = preve[v]
                if pv < 0:
                    addf = 0
                    break
                e = g[pv][ei]
                if e.cap < addf:
                    addf = e.cap
                    if addf == 0:
                        break
                v = pv
            if addf == 0:
                break

            v = t
            while v != s:
                pv = prevv[v]
                ei = preve[v]
                e = g[pv][ei]
                e.cap -= addf
                g[v][e.rev].cap += addf
                cost += addf * e.cost
                v = pv

            flow += addf

        return flow, cost

class Solver:
    def solve(self, problem: dict[str, Any], **kwargs: Any) -> list[int]:
        T = int(problem["T"])
        N = int(problem["N"])
        if T <= 0:
            return []
        if N <= 0:
            return [-1] * T

        prices = np.asarray(problem["prices"], dtype=np.float64)
        capacities = np.asarray(problem["capacities"], dtype=np.int64)
        probs = np.asarray(problem["probs"], dtype=np.float64)  # (T, N)

        # Shrink products with zero capacity.
        active_mask = capacities > 0
        if not np.any(active_mask):
            return [-1] * T

        active_idx = np.nonzero(active_mask)[0]
        cap = capacities[active_idx].astype(np.int64, copy=False)
        cap = np.minimum(cap, T)
        n_active = int(active_idx.size)

        # Expected revenues for active products.
        w = probs[:, active_idx] * prices[active_idx]  # (T, n_active)

        # -------- Fast exact path: reduce to rectangular assignment --------
        # Expand each product i into cap[i] identical "copies" (columns).
        # Each period (row) assigned to at most one copy; unused copies allowed.
        # If total copies < T, add dummy idle columns with 0 revenue.
        total_copies = int(np.sum(cap))
        if total_copies == 0:
            return [-1] * T

        # Decide whether matrix expansion is safe (memory/time).
        # The cost matrix is float64 of size T * M.
        dummy = max(0, T - total_copies)
        M = total_copies + dummy
        # Heuristic safety limit (~160MB for float64 at 20e6 entries).
        if M >= T and (T * M) <= 20_000_000:
            # Build mapping column -> active-product-index.
            col_prod = np.repeat(np.arange(n_active, dtype=np.int32), cap.astype(np.int64))
            # Build cost matrix: minimize -revenue (idle columns have 0 cost).
            cost = -w[:, col_prod]  # shape (T, total_copies), materialized
            if dummy:
                cost = np.concatenate((cost, np.zeros((T, dummy), dtype=np.float64)), axis=1)

            row_ind, col_ind = linear_sum_assignment(cost)

            offer = [-1] * T
            # row_ind is typically 0..T-1, but don't assume.
            for r, c in zip(row_ind.tolist(), col_ind.tolist()):
                if c < total_copies:
                    offer[r] = int(active_idx[int(col_prod[c])])
                else:
                    offer[r] = -1
            return offer

        # -------- Fallback exact path: float min-cost flow --------
        # Node layout:
        # 0: source
        # 1..T: periods
        # T+1..T+n_active: products
        # T+n_active+1: sink
        source = 0
        period0 = 1
        prod0 = period0 + T
        sink = prod0 + n_active
        V = sink + 1

        mcf = _MinCostFlow(V)

        for t in range(T):
            mcf.add_edge(source, period0 + t, 1, 0.0)

        for j in range(n_active):
            mcf.add_edge(prod0 + j, sink, int(cap[j]), 0.0)

        C = float(np.max(w)) if w.size else 0.0
        if C < 0.0:
            C = 0.0

        for t in range(T):
            pt = period0 + t
            wt = w[t]
            mcf.add_edge(pt, sink, 1, C)
            base_to = prod0
            g_add = mcf.add_edge
            for j in range(n_active):
                g_add(pt, base_to + j, 1, C - float(wt[j]))

        flow, _ = mcf.min_cost_flow(source, sink, T)
        if flow != T:
            return [-1] * T

        offer = [-1] * T
        g = mcf.g
        for t in range(T):
            pt = period0 + t
            chosen = -1
            for e in g[pt]:
                if e.cap == 0:
                    to = e.to
                    if prod0 <= to < sink:
                        chosen = int(active_idx[to - prod0])
                    else:
                        chosen = -1
                    break
            offer[t] = chosen

        return offer