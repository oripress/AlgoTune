from __future__ import annotations

from collections import deque
from typing import Any, List, Optional

try:
    from pysat.solvers import Solver as _SatSolver  # type: ignore
except Exception:  # pragma: no cover
    _SatSolver = None  # type: ignore

def _greedy_upper_bound(neigh: List[int], alive: int) -> List[int]:
    """Greedy vertex cover by repeatedly taking max-degree vertex."""
    cover: List[int] = []
    a = alive
    while a:
        best_v = -1
        best_deg = 0
        aa = a
        while aa:
            b = aa & -aa
            v = b.bit_length() - 1
            d = (neigh[v] & a).bit_count()
            if d > best_deg:
                best_deg = d
                best_v = v
            aa &= aa - 1
        if best_deg == 0:
            break
        cover.append(best_v)
        a &= ~(1 << best_v)
    return cover

def _matching_lower_bound(neigh: List[int], alive: int) -> int:
    """Greedy maximal matching size (lower bound for vertex cover)."""
    m = 0
    a = alive
    while a:
        vb = a & -a
        v = vb.bit_length() - 1
        a ^= vb
        nb = neigh[v] & a
        if nb:
            ub = nb & -nb
            a ^= ub
            m += 1
    return m

class _BBState:
    __slots__ = ("neigh", "best_size", "best_cover", "cover")

    def __init__(self, neigh: List[int], best_size: int, best_cover: List[int], cover: List[int]):
        self.neigh = neigh
        self.best_size = best_size
        self.best_cover = best_cover
        self.cover = cover

def _bb_solve_min_vc(neigh: List[int]) -> List[int]:
    n = len(neigh)
    alive0 = (1 << n) - 1

    iso = 0
    for i in range(n):
        if neigh[i] == 0:
            iso |= 1 << i
    alive0 &= ~iso
    if alive0 == 0:
        return []

    ub_cover = _greedy_upper_bound(neigh, alive0)
    state = _BBState(neigh=neigh, best_size=len(ub_cover), best_cover=ub_cover, cover=[])

    def reduce_graph(alv: int) -> int:
        """Apply safe reductions; mutates state.cover (caller must backtrack)."""
        neigh_local = state.neigh
        cover_local = state.cover

        while True:
            changed = False

            # Remove isolated vertices
            a = alv
            while a:
                vb = a & -a
                v = vb.bit_length() - 1
                if (neigh_local[v] & alv) == 0:
                    alv &= ~vb
                    changed = True
                a &= a - 1

            # Degree-1 rule (apply one at a time, restart)
            a = alv
            forced = False
            while a:
                vb = a & -a
                v = vb.bit_length() - 1
                nb = neigh_local[v] & alv
                if nb and (nb & (nb - 1)) == 0:
                    u_bit = nb
                    u = u_bit.bit_length() - 1
                    cover_local.append(u)
                    alv &= ~((1 << v) | u_bit)
                    changed = True
                    forced = True
                    break
                a &= a - 1
            if forced:
                continue

            # High-degree forced inclusion to beat incumbent
            budget = state.best_size - len(cover_local)
            if budget >= 0:
                a = alv
                while a:
                    vb = a & -a
                    v = vb.bit_length() - 1
                    if (neigh_local[v] & alv).bit_count() > budget:
                        cover_local.append(v)
                        alv &= ~vb
                        changed = True
                        forced = True
                        break
                    a &= a - 1
                if forced:
                    continue

            if not changed:
                return alv

    def dfs(alv: int) -> None:
        cover_local = state.cover
        if len(cover_local) >= state.best_size:
            return

        base_len = len(cover_local)
        try:
            alv = reduce_graph(alv)
            if len(cover_local) >= state.best_size:
                return

            lb = len(cover_local) + _matching_lower_bound(state.neigh, alv)
            if lb >= state.best_size:
                return

            best_v = -1
            best_deg = 0
            best_nb = 0
            a = alv
            neigh_local = state.neigh
            while a:
                vb = a & -a
                v = vb.bit_length() - 1
                nb = neigh_local[v] & alv
                deg = nb.bit_count()
                if deg > best_deg:
                    best_deg = deg
                    best_v = v
                    best_nb = nb
                a &= a - 1

            if best_deg == 0:
                if len(cover_local) < state.best_size:
                    state.best_size = len(cover_local)
                    state.best_cover = cover_local.copy()
                return

            v_bit = 1 << best_v

            # Branch 1: include v
            cover_local.append(best_v)
            dfs(alv & ~v_bit)
            cover_local.pop()

            # Branch 2: exclude v => include all neighbors
            nbrs = best_nb
            cnt = nbrs.bit_count()
            if len(cover_local) + cnt < state.best_size:
                x = nbrs
                while x:
                    ub = x & -x
                    cover_local.append(ub.bit_length() - 1)
                    x &= x - 1
                dfs(alv & ~(nbrs | v_bit))
                del cover_local[-cnt:]
        finally:
            del cover_local[base_len:]

    dfs(alive0)
    return state.best_cover

def _bipartite_min_vc(adj: List[List[int]]) -> Optional[List[int]]:
    """If graph is bipartite, return exact min vertex cover (Kőnig). Else None."""
    n = len(adj)
    side = [-1] * n
    q: deque[int] = deque()

    for s in range(n):
        if side[s] != -1:
            continue
        side[s] = 0
        q.append(s)
        while q:
            u = q.popleft()
            su = side[u]
            for v in adj[u]:
                sv = side[v]
                if sv == -1:
                    side[v] = su ^ 1
                    q.append(v)
                elif sv == su:
                    return None

    U = [i for i in range(n) if side[i] == 0]
    V = [i for i in range(n) if side[i] == 1]
    if not U or not V:
        return []

    posU = [-1] * n
    posV = [-1] * n
    for i, u in enumerate(U):
        posU[u] = i
    for i, v in enumerate(V):
        posV[v] = i

    adjU: List[List[int]] = [[] for _ in range(len(U))]
    for ui, u in enumerate(U):
        lst = adjU[ui]
        for w in adj[u]:
            pv = posV[w]
            if pv != -1:
                lst.append(pv)

    # Hopcroft-Karp
    nu = len(U)
    nv = len(V)
    pairU = [-1] * nu
    pairV = [-1] * nv
    dist = [0] * nu
    INF = 10**9

    def bfs() -> bool:
        dq = deque()
        found = False
        for u in range(nu):
            if pairU[u] == -1:
                dist[u] = 0
                dq.append(u)
            else:
                dist[u] = INF
        while dq:
            u = dq.popleft()
            du = dist[u] + 1
            for v in adjU[u]:
                pu = pairV[v]
                if pu == -1:
                    found = True
                elif dist[pu] == INF:
                    dist[pu] = du
                    dq.append(pu)
        return found

    def dfs(u: int) -> bool:
        for v in adjU[u]:
            pu = pairV[v]
            if pu == -1 or (dist[pu] == dist[u] + 1 and dfs(pu)):
                pairU[u] = v
                pairV[v] = u
                return True
        dist[u] = INF
        return False

    while bfs():
        for u in range(nu):
            if pairU[u] == -1:
                dfs(u)

    # Min vertex cover from alternating reachability
    visU = [False] * nu
    visV = [False] * nv
    dq = deque()
    for u in range(nu):
        if pairU[u] == -1:
            visU[u] = True
            dq.append(u)

    while dq:
        u = dq.popleft()
        mu = pairU[u]
        for v in adjU[u]:
            if v == mu:
                continue
            if not visV[v]:
                visV[v] = True
                pu = pairV[v]
                if pu != -1 and not visU[pu]:
                    visU[pu] = True
                    dq.append(pu)

    cover: List[int] = []
    for ui, u in enumerate(U):
        if not visU[ui]:
            cover.append(u)
    for vi, v in enumerate(V):
        if visV[vi]:
            cover.append(v)
    return cover

def _pysat_min_vc_from_neigh(neigh: List[int], ub: int) -> List[int]:
    """Exact MVC using an incremental SAT solver (Minicard) with monotone tightening."""
    if _SatSolver is None:
        return _bb_solve_min_vc(neigh)

    n = len(neigh)
    lits = list(range(1, n + 1))

    # Build edge clauses
    clauses: List[List[int]] = []
    for i in range(n):
        nb = neigh[i] & ~((1 << (i + 1)) - 1)  # j > i
        li = i + 1
        while nb:
            b = nb & -nb
            j = b.bit_length() - 1
            clauses.append([li, j + 1])
            nb ^= b

    best: List[int] = list(range(n))
    best_k = n

    with _SatSolver(name="Minicard") as s:
        s.append_formula(clauses)

        # Add an initial atmost constraint; if ub is n, skip it.
        if ub < n:
            if hasattr(s, "add_atmost"):
                s.add_atmost(lits=lits, k=ub)
            else:
                from pysat.card import CardEnc, EncType  # type: ignore

                s.append_formula(CardEnc.atmost(lits=lits, bound=ub, encoding=EncType.seqcounter).clauses)

        while s.solve():
            model = s.get_model()
            # Count selected among first n vars
            k = 0
            sel: List[int] = []
            for i in range(n):
                if model[i] > 0:
                    k += 1
                    sel.append(i)
            best = sel
            best_k = k
            if k == 0:
                return []
            nbnd = k - 1
            if nbnd < 0:
                break
            if hasattr(s, "add_atmost"):
                s.add_atmost(lits=lits, k=nbnd)
            else:
                from pysat.card import CardEnc, EncType  # type: ignore

                s.append_formula(CardEnc.atmost(lits=lits, bound=nbnd, encoding=EncType.seqcounter).clauses)

    return best if best_k < n else list(range(n))

class Solver:
    def solve(self, problem: List[List[int]], **kwargs: Any) -> Any:
        n = len(problem)
        if n == 0:
            return []

        neigh = [0] * n
        adj: List[List[int]] = [[] for _ in range(n)]

        edge_found = False
        m = 0
        for i in range(n):
            row = problem[i]
            ni = 0
            ai = adj[i]
            for j in range(i + 1, n):
                if row[j]:
                    bitj = 1 << j
                    ni |= bitj
                    neigh[j] |= 1 << i
                    ai.append(j)
                    adj[j].append(i)
                    edge_found = True
                    m += 1
            neigh[i] |= ni

        if not edge_found:
            return []

        # Fast exact path for bipartite graphs
        cover_bi = _bipartite_min_vc(adj)
        if cover_bi is not None:
            return cover_bi

        # Upper bound from greedy for SAT tightening / pruning
        alive = (1 << n) - 1
        ub = len(_greedy_upper_bound(neigh, alive))

        # Heuristic routing:
        # - Dense/larger graphs: incremental SAT in C is usually faster than Python B&B.
        # - Smaller/sparser graphs: Python B&B is very fast.
        if _SatSolver is not None:
            # density in [0,1]
            dens_num = 2 * m
            dens_den = n * (n - 1) if n > 1 else 1
            if n >= 90 or dens_num * 10 >= dens_den * 4:  # n>=90 or density >= 0.4
                return _pysat_min_vc_from_neigh(neigh, ub)

        return _bb_solve_min_vc(neigh)