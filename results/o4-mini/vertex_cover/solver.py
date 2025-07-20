import sys
from ortools.sat.python import cp_model

sys.setrecursionlimit(10000)

class Solver:
    def solve(self, problem, **kwargs):
        n = len(problem)
        if n == 0:
            return []

        # Small-to-medium graphs: use Cython-accelerated max_clique on complement for MIS
        try:
            import graph_mis
            use_cy = True
        except ImportError:
            use_cy = False
        if n <= 80:
            # Build bitsets for adjacency and complement-adjacency
            full = (1 << n) - 1
            adj = [0] * n
            for i, row in enumerate(problem):
                bits = 0
                for j, v in enumerate(row):
                    if v:
                        bits |= 1 << j
                adj[i] = bits
            comp_adj = [(full ^ adj[i]) & ~(1 << i) for i in range(n)]
            if use_cy:
                best_mask = graph_mis.max_clique_mask(comp_adj)
            else:
                # Fallback Python greedy-coloring BK for MIS
                best_mask = 0
                best_size = 0
                def bk(R_mask, P_list):
                    nonlocal best_mask, best_size
                    R_size = R_mask.bit_count()
                    # greedy coloring to get upper bounds
                    colors = {}
                    color_class = []
                    for v in P_list:
                        placed = False
                        for ci, cl_bits in enumerate(color_class):
                            if not (comp_adj[v] & cl_bits):
                                color_class[ci] |= 1 << v
                                colors[v] = ci + 1
                                placed = True
                                break
                        if not placed:
                            color_class.append(1 << v)
                            colors[v] = len(color_class)
                    # vertices sorted by increasing color
                    P_sorted = sorted(P_list, key=lambda u: colors[u])
                    # branch in reverse order
                    for v in reversed(P_sorted):
                        c = colors[v]
                        if R_size + c <= best_size:
                            return
                        v_bit = 1 << v
                        R_new = R_mask | v_bit
                        neighbors = comp_adj[v]
                        P_new = [u for u in P_sorted if neighbors & (1 << u)]
                        if not P_new:
                            size_new = R_new.bit_count()
                            if size_new > best_size:
                                best_size = size_new
                                best_mask = R_new
                        else:
                            bk(R_new, P_new)
                    return
                # start search
                bk(0, list(range(n)))
            # convert MIS to vertex cover and return
            cover = [i for i in range(n) if not ((best_mask >> i) & 1)]
            return cover
        # Fallback to CP-SAT for larger n with reduction
        active = set(range(n))
        neighbors = {i: set(j for j, v in enumerate(problem[i]) if v) for i in active}
        forced = set()
        changed = True
        while changed:
            changed = False
            # Remove degree-0 vertices
            to_remove = [u for u in active if not neighbors[u]]
            if to_remove:
                for u in to_remove:
                    active.remove(u)
                    neighbors.pop(u, None)
                changed = True
                continue
            # Remove degree-1 vertices (leaf reduction)
            deg1 = [u for u in active if len(neighbors[u]) == 1]
            if deg1:
                for u in deg1:
                    if u not in active:
                        continue
                    v = next(iter(neighbors[u]))
                    forced.add(v)
                    # remove u and v
                    for w in list(neighbors.get(v, [])):
                        neighbors[w].remove(v)
                    neighbors.pop(v, None)
                    for w in list(neighbors.get(u, [])):
                        neighbors[w].remove(u)
                    neighbors.pop(u, None)
                    active.discard(u)
                    active.discard(v)
                changed = True
                continue

        # If all vertices removed, return forced covers
        if not active:
            return sorted(forced)
        # If any reduction occurred, solve reduced subproblem
        if len(active) < n or forced:
            mapping = {old: idx for idx, old in enumerate(sorted(active))}
            inv_mapping = {idx: old for old, idx in mapping.items()}
            m = len(active)
            sub_problem = [[0] * m for _ in range(m)]
            for old_i in active:
                i2 = mapping[old_i]
                for old_j in neighbors.get(old_i, []):
                    if old_j in active:
                        j2 = mapping[old_j]
                        sub_problem[i2][j2] = 1
                        sub_problem[j2][i2] = 1
            # Recursively solve subproblem
            sub_cover = self.solve(sub_problem)
            result = set(forced)
            for new_i in sub_cover:
                result.add(inv_mapping[new_i])
            return sorted(result)
        # Check if the remaining graph is bipartite and solve via matching if so
        from collections import deque
        color = {}
        is_bipartite = True
        for u in active:
            if u not in color:
                color[u] = 0
                queue = deque([u])
                while queue and is_bipartite:
                    v = queue.popleft()
                    for w in neighbors[v]:
                        if w not in active:
                            continue
                        if w not in color:
                            color[w] = color[v] ^ 1
                            queue.append(w)
                        elif color[w] == color[v]:
                            is_bipartite = False
                            break
        if is_bipartite:
            # build bipartition
            U = [u for u in active if color[u] == 0]
            V = [u for u in active if color[u] == 1]
            adjU = {u: [w for w in neighbors[u] if w in V] for u in U}
            # Hopcroft-Karp for max matching
            INF = len(active) + 1
            dist = {}
            pair_U = {u: None for u in U}
            pair_V = {v: None for v in V}
            def bfs_hk():
                dq = deque()
                for uu in U:
                    if pair_U[uu] is None:
                        dist[uu] = 0
                        dq.append(uu)
                    else:
                        dist[uu] = INF
                dist[None] = INF
                while dq:
                    uu = dq.popleft()
                    if dist[uu] < dist[None]:
                        for vv in adjU[uu]:
                            pu = pair_V[vv]
                            if pu is None:
                                if dist[None] == INF:
                                    dist[None] = dist[uu] + 1
                            else:
                                if dist[pu] == INF:
                                    dist[pu] = dist[uu] + 1
                                    dq.append(pu)
                return dist[None] != INF
            def dfs_hk(uu):
                for vv in adjU[uu]:
                    pu = pair_V[vv]
                    if pu is None or (dist[pu] == dist[uu] + 1 and dfs_hk(pu)):
                        pair_U[uu] = vv
                        pair_V[vv] = uu
                        return True
                dist[uu] = INF
                return False
            matching = 0
            while bfs_hk():
                for uu in U:
                    if pair_U[uu] is None and dfs_hk(uu):
                        matching += 1
            # König's theorem to recover minimum vertex cover
            visited_U = set()
            visited_V = set()
            dq2 = deque([('U', uu) for uu in U if pair_U[uu] is None])
            for side, uu in list(dq2):
                visited_U.add(uu)
            while dq2:
                side, xx = dq2.popleft()
                if side == 'U':
                    for vv in adjU[xx]:
                        if pair_U[xx] != vv and vv not in visited_V:
                            visited_V.add(vv)
                            dq2.append(('V', vv))
                else:
                    uu2 = pair_V[xx]
                    if uu2 is not None and uu2 not in visited_U:
                        visited_U.add(uu2)
                        dq2.append(('U', uu2))
            cover = set(u for u in U if u not in visited_U) | set(v for v in V if v in visited_V)
            cover.update(forced)
            return sorted(cover)
        # No reduction possible, solve full problem via CP-SAT
        model = cp_model.CpModel()
        x = [model.NewBoolVar(f"x{i}") for i in range(n)]
        for i in range(n):
            for j in range(i + 1, n):
                if problem[i][j]:
                    model.Add(x[i] + x[j] >= 1)
        model.Minimize(sum(x))
        # approximate vertex cover for CP-SAT hint
        approx_neighbors = {u: set(neighbors[u]) for u in active}
        approx_cover = []
        while True:
            found_edge = False
            for u in approx_neighbors:
                if approx_neighbors[u]:
                    v = next(iter(approx_neighbors[u]))
                    approx_cover.append(u)
                    # remove edges incident to u
                    for w in list(approx_neighbors[u]):
                        approx_neighbors[w].remove(u)
                    approx_neighbors[u].clear()
                    found_edge = True
                    break
            if not found_edge:
                break
        for i in range(n):
            if i in approx_cover:
                model.AddHint(x[i], 1)
            else:
                model.AddHint(x[i], 0)
        solver = cp_model.CpSolver()
        solver.parameters.num_search_workers = 8
        status = solver.Solve(model)
        if status in (cp_model.OPTIMAL, cp_model.FEASIBLE):
            base_cover = [i for i in range(n) if solver.Value(x[i]) == 1]
            return sorted(set(base_cover) | forced)
        # As a fallback, include all vertices
        return sorted(set(range(n)) | forced)