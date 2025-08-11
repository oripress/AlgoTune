from __future__ import annotations

from typing import Any, List, Optional
import bisect
import math
from collections import deque

try:
    # python-sat
    from pysat.solvers import Solver as SATSolver
except Exception as e:  # pragma: no cover
    SATSolver = None  # type: ignore

class Solver:
    def solve(self, problem: List[List[float]], **kwargs) -> List[int]:
        """
        Solve the Bottleneck Traveling Salesman Problem (BTSP).

        Given a symmetric positive distance matrix 'problem', returns a tour as a list of
        city indices starting at city 0 and ending at city 0 (i.e., a Hamiltonian cycle),
        such that the maximum edge length along the tour is minimized.

        This implementation uses an exact SAT-based encoding with binary search
        over the bottleneck threshold and several optimizations to minimize SAT size.
        """
        n = len(problem)
        # Handle trivial cases
        if n <= 0:
            return []
        if n == 1:
            return [0, 0]
        if n == 2:
            # The only Hamiltonian cycle is 0 -> 1 -> 0
            return [0, 1, 0]

        if SATSolver is None:
            # Fallback: simple cycle (0..n-1..0). This may not be optimal but avoids crash.
            # However, in the intended environment python-sat is available.
            tour = list(range(n)) + [0]
            return tour

        # Precompute unique sorted edge weights (upper triangle)
        weights = set()
        for i in range(n):
            row = problem[i]
            for j in range(i + 1, n):
                w = row[j]
                if math.isfinite(w):
                    weights.add(float(w))
        if not weights:
            # no edges? invalid, return trivial
            return list(range(n)) + [0]

        wlist = sorted(weights)

        # Precompute per-node neighbors sorted by weight once
        neigh_w: List[List[float]] = [[] for _ in range(n)]
        neigh_j: List[List[int]] = [[] for _ in range(n)]
        for i in range(n):
            row = problem[i]
            pairs = [(row[j], j) for j in range(n) if j != i]
            pairs.sort(key=lambda x: x[0])
            if pairs:
                neigh_w[i] = [p[0] for p in pairs]
                neigh_j[i] = [p[1] for p in pairs]

        # Lower bounds to prune the search space
        # 1) MST bottleneck (max edge in an MST) provides a lower bound.
        lb_mst = self._mst_bottleneck(problem)

        # 2) Degree-based bound: to be in a Hamiltonian cycle (n >= 3),
        # each node must have at least two neighbors under threshold.
        # Hence, threshold must be at least the second-smallest distance from each node.
        if n >= 3:
            deg_lb = 0.0
            for i in range(n):
                wl = neigh_w[i]
                if len(wl) >= 2:
                    second = wl[1]
                else:
                    second = float("inf")
                if second > deg_lb:
                    deg_lb = second
            lower_bound = max(lb_mst, deg_lb)
        else:
            lower_bound = lb_mst

        # Upper bound via a greedy nearest-neighbor tour
        ub_val = self._greedy_upper_bound(problem)
        # Restrict search range using bounds
        lo = bisect.bisect_left(wlist, lower_bound)
        hi = bisect.bisect_right(wlist, ub_val) - 1
        if lo >= len(wlist):
            lo = len(wlist) - 1
        if hi < 0:
            hi = 0
        if lo > hi:
            # Safety: if bounds crossed due to numeric issues
            lo, hi = hi, lo

        best_order: Optional[List[int]] = None
        # Binary search on threshold
        while lo <= hi:
            mid = (lo + hi) // 2
            thr = wlist[mid]
            order = self._hamiltonian_cycle_order_under_threshold(thr, neigh_w, neigh_j)
            if order is not None:
                best_order = order
                hi = mid - 1
            else:
                lo = mid + 1

        if best_order is None:
            # Should not happen in complete graphs; fallback to trivial cycle
            tour = list(range(n)) + [0]
            return tour

        # Ensure the tour starts at 0
        # Our encoding pins 0 at position 0 already, but rotate defensively if needed.
        try:
            start_idx = best_order.index(0)
        except ValueError:
            start_idx = 0
        if start_idx != 0:
            best_order = best_order[start_idx:] + best_order[:start_idx]

        best_order.append(best_order[0])
        return best_order

    @staticmethod
    def _mst_bottleneck(problem: List[List[float]]) -> float:
        """Compute the maximum edge weight in a Prim's MST. O(n^2)."""
        n = len(problem)
        if n <= 1:
            return 0.0
        in_tree = [False] * n
        min_edge = [float("inf")] * n
        min_edge[0] = 0.0
        max_in_mst = 0.0
        for _ in range(n):
            # Select the non-tree vertex with minimal connecting edge
            u = -1
            best = float("inf")
            for v in range(n):
                if not in_tree[v] and min_edge[v] < best:
                    best = min_edge[v]
                    u = v
            if u == -1:
                break  # disconnected? shouldn't happen for complete graph
            in_tree[u] = True
            if best > max_in_mst:
                max_in_mst = best
            # Update edges
            pu = problem[u]
            for v in range(n):
                if not in_tree[v]:
                    w = pu[v]
                    if w < min_edge[v]:
                        min_edge[v] = w
        return max_in_mst

    @staticmethod
    def _greedy_upper_bound(problem: List[List[float]]) -> float:
        """Construct a greedy tour (nearest neighbor) to get an upper bound on bottleneck."""
        n = len(problem)
        visited = [False] * n
        order = [0]
        visited[0] = True
        cur = 0
        for _ in range(1, n):
            best_j = -1
            best_w = float("inf")
            row = problem[cur]
            for j in range(n):
                if not visited[j] and j != cur:
                    w = row[j]
                    if w < best_w:
                        best_w = w
                        best_j = j
            if best_j == -1:
                # complete
                break
            order.append(best_j)
            visited[best_j] = True
            cur = best_j
        # Close the tour
        order.append(0)
        # Compute bottleneck
        b = 0.0
        for u, v in zip(order[:-1], order[1:]):
            w = problem[u][v]
            if w > b:
                b = w
        return b

    def _hamiltonian_cycle_order_under_threshold(
        self, thr: float, neigh_w: List[List[float]], neigh_j: List[List[int]]
    ) -> Optional[List[int]]:
        """
        Check feasibility of a Hamiltonian cycle with all edges <= thr.
        First try fast perfect matching + cycle patching; if it fails, fall back to SAT.
        Returns the order of vertices (length n) if feasible (0..n-1), else None.
        """
        n = len(neigh_w)

        # Build adjacency bitmasks under threshold
        nb = [0] * n  # nb[i] bitmask of neighbors of i allowed
        for i in range(n):
            wl = neigh_w[i]
            jl = neigh_j[i]
            # number of neighbors with weight <= thr
            cnt = bisect.bisect_right(wl, thr)
            m = 0
            for t in range(cnt):
                m |= 1 << jl[t]
            # forbid self-loops
            m &= ~ (1 << i)
            nb[i] = m

        # Degree feasibility checks
        if n >= 3:
            # Need at least two neighbors for each city
            for i in range(n):
                if (nb[i] & (nb[i] - 1)) == 0:  # popcount < 2
                    return None
        else:
            for i in range(n):
                if nb[i] == 0:
                    return None

        # Connectivity check via BFS
        seen_mask = 1 << 0
        dq = deque([0])
        while dq:
            u = dq.popleft()
            m = nb[u]
            while m:
                b = m & -m
                v = b.bit_length() - 1
                m -= b
                if not (seen_mask >> v) & 1:
                    seen_mask |= 1 << v
                    dq.append(v)
        if seen_mask != (1 << n) - 1:
            return None

        # Fast perfect matching on bipartite graph (left U = cities, right V = cities),
        # with edges for allowed adjacencies (i -> j), i != j.
        # Use Hopcroft–Karp to find permutation succ (size n) if possible.
        adj: List[List[int]] = [[] for _ in range(n)]
        for i in range(n):
            m = nb[i]
            while m:
                b = m & -m
                j = b.bit_length() - 1
                m -= b
                adj[i].append(j)

        # Hopcroft–Karp
        INF = 10**9
        pairU = [-1] * n
        pairV = [-1] * n
        dist = [INF] * n

        from collections import deque as _dq

        def bfs_hk() -> bool:
            q = _dq()
            for u in range(n):
                if pairU[u] == -1:
                    dist[u] = 0
                    q.append(u)
                else:
                    dist[u] = INF
            found_free = False
            while q:
                u = q.popleft()
                du = dist[u]
                for v in adj[u]:
                    pu = pairV[v]
                    if pu == -1:
                        found_free = True
                    else:
                        if dist[pu] == INF:
                            dist[pu] = du + 1
                            q.append(pu)
            return found_free

        def dfs_hk(u: int) -> bool:
            for v in adj[u]:
                pu = pairV[v]
                if pu == -1 or (dist[pu] == dist[u] + 1 and dfs_hk(pu)):
                    pairU[u] = v
                    pairV[v] = u
                    return True
            dist[u] = INF
            return False

        matching = 0
        while bfs_hk():
            for u in range(n):
                if pairU[u] == -1:
                    if dfs_hk(u):
                        matching += 1

        if matching == n:
            succ = pairU[:]  # succ[u] = v
            pred = pairV[:]  # pred[v] = u

            # Check for single cycle
            visited = [False] * n
            cycles = []
            for i in range(n):
                if not visited[i]:
                    cur = i
                    cyc = []
                    while not visited[cur]:
                        visited[cur] = True
                        cyc.append(cur)
                        cur = succ[cur]
                    cycles.append(cyc)
            if len(cycles) == 1:
                # Build order starting at 0
                order = [0] * n
                order[0] = 0
                cur = succ[0]
                idx = 1
                while cur != 0:
                    order[idx] = cur
                    cur = succ[cur]
                    idx += 1
                return order

            # Try to patch cycles via 2-edge exchanges until one cycle remains
            # Utility to recompute cycles mapping
            def recompute_cycles() -> List[List[int]]:
                visited2 = [False] * n
                cyc_list: List[List[int]] = []
                for s in range(n):
                    if not visited2[s]:
                        cur = s
                        cyc = []
                        while not visited2[cur]:
                            visited2[cur] = True
                            cyc.append(cur)
                            cur = succ[cur]
                        cyc_list.append(cyc)
                return cyc_list

            def merge_attempt() -> bool:
                # map node -> cycle id
                cyc_list = recompute_cycles()
                if len(cyc_list) <= 1:
                    return True
                cid = [-1] * n
                for idx, cyc in enumerate(cyc_list):
                    for u in cyc:
                        cid[u] = idx
                # Try all vertices, find cross-cycle patch
                for a in range(n):
                    ca = cid[a]
                    x = succ[a]
                    ma = nb[a]
                    while ma:
                        bbit = ma & -ma
                        b = bbit.bit_length() - 1
                        ma -= bbit
                        if cid[b] == ca:
                            continue  # same cycle
                        pb = pred[b]
                        # Check if we can redirect pb -> x
                        if (nb[pb] >> x) & 1:
                            # Patch: a->b and pb->x
                            # Remove a->x and pb->b
                            succ[a] = b
                            pred[b] = a
                            succ[pb] = x
                            pred[x] = pb
                            return True
                return False

            # Perform up to n-1 merges
            merged = True
            for _ in range(n - 1):
                if merged:
                    merged = merge_attempt()
                else:
                    break
                # Early stop if single cycle
                # Quick check: follow from 0 for n steps and see if returns to 0 without repeat
                v = 0
                for _t in range(n):
                    v = succ[v]
                if v == 0:
                    # likely a single cycle (not rigorous), confirm
                    vis = [False] * n
                    cur = 0
                    for _t in range(n):
                        if vis[cur]:
                            break
                        vis[cur] = True
                        cur = succ[cur]
                    if cur == 0 and all(vis):
                        order = [0] * n
                        order[0] = 0
                        cur = succ[0]
                        idx = 1
                        while cur != 0:
                            order[idx] = cur
                            cur = succ[cur]
                            idx += 1
                        return order

        # Fallback to SAT with minimal domains (position-based) to reduce size
        if SATSolver is None:
            return None
        try:
            solver = SATSolver(name="m22")
        except Exception:
            solver = SATSolver(name="g3")

        # Minimal domains:
        # pos 0 -> {0}, pos 1 -> neighbors of 0, pos n-1 -> nodes with edge to 0, others -> all except 0
        pos_dom = [0] * n
        pos_dom[0] = 1 << 0
        pos_dom[1] = nb[0]
        last_mask = 0
        for i in range(1, n):
            if nb[i] & 1:
                last_mask |= 1 << i
        pos_dom[n - 1] = last_mask
        nonzero_mask = ((1 << n) - 1) ^ 1
        for k in range(2, n - 1):
            pos_dom[k] = nonzero_mask

        # Quick infeasibility
        if pos_dom[1] == 0 or pos_dom[n - 1] == 0:
            solver.delete()
            return None

        # Build allowed_pos_lists
        allowed_pos_lists: List[List[int]] = []
        for k in range(n):
            if k == 0:
                allowed_pos_lists.append([0])
                continue
            lst = []
            m = pos_dom[k]
            while m:
                b = m & -m
                i = b.bit_length() - 1
                m -= b
                lst.append(i)
            allowed_pos_lists.append(lst)

        idmap: dict[tuple[int, int], int] = {}
        next_var = 1

        def var(i: int, k: int) -> int:
            nonlocal next_var
            key = (i, k)
            v = idmap.get(key)
            if v is None:
                v = next_var
                next_var += 1
                idmap[key] = v
            return v

        def new_var() -> int:
            nonlocal next_var
            v = next_var
            next_var += 1
            return v

        def add_amo_sequential(vars_grp: List[int]) -> None:
            m = len(vars_grp)
            if m <= 1:
                return
            if m == 2:
                solver.add_clause([-vars_grp[0], -vars_grp[1]])
                return
            s_vars = [new_var() for _ in range(m - 1)]
            solver.add_clause([-vars_grp[0], s_vars[0]])
            for i in range(1, m - 1):
                xi = vars_grp[i]
                si = s_vars[i]
                sim1 = s_vars[i - 1]
                solver.add_clause([-xi, si])
                solver.add_clause([-sim1, si])
                solver.add_clause([-xi, -sim1])
            solver.add_clause([-vars_grp[-1], -s_vars[-1]])

        # Fix city 0 at position 0
        solver.add_clause([var(0, 0)])

        # Position exactly-one constraints (k = 1..n-1)
        for k in range(1, n):
            cities = allowed_pos_lists[k]
            if not cities:
                solver.delete()
                return None
            vars_k = [var(i, k) for i in cities]
            solver.add_clause(vars_k)
            add_amo_sequential(vars_k)

        # City exactly-one constraints (i = 1..n-1)
        city_pos_lists: List[List[int]] = [[] for _ in range(n)]
        for k in range(1, n):
            for i in allowed_pos_lists[k]:
                city_pos_lists[i].append(k)
        for i in range(1, n):
            pos_list = city_pos_lists[i]
            if not pos_list:
                solver.delete()
                return None
            vars_i = [var(i, k) for k in pos_list]
            solver.add_clause(vars_i)
            add_amo_sequential(vars_i)

        # Adjacency constraints for k = 0..n-2
        for k in range(0, n - 1):
            kp1 = k + 1
            next_cities = allowed_pos_lists[kp1]
            if k == 0:
                i_iter = (0,)
            else:
                i_iter = allowed_pos_lists[k]
            for i in i_iter:
                clause = [-var(i, k)]
                for j in next_cities:
                    if (nb[i] >> j) & 1:
                        clause.append(var(j, kp1))
                if len(clause) == 1:
                    solver.add_clause(clause)
                else:
                    solver.add_clause(clause)

        sat = solver.solve()
        if not sat:
            solver.delete()
            return None
        model = solver.get_model()
        solver.delete()
        true_set = set(l for l in model if l > 0)

        # Decode order: position 0 is 0; for k>=1 find the single true var
        order = [0] * n
        for k in range(1, n):
            found = -1
            for i in allowed_pos_lists[k]:
                if var(i, k) in true_set:
                    found = i
                    break
            if found < 0:
                return None
            order[k] = found

        if len(set(order)) != n:
            return None
        return order