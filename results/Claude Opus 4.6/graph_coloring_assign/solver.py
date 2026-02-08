from ortools.sat.python import cp_model
import networkx as nx
from networkx.algorithms.approximation import clique as approx_clique

try:
    from pysat.solvers import Glucose4
    HAS_PYSAT = True
except ImportError:
    HAS_PYSAT = False


class Solver:
    def solve(self, problem, **kwargs):
        n = len(problem)
        if n == 0:
            return []
        if n == 1:
            return [1]

        # Build adjacency
        edges = []
        adj = [[] for _ in range(n)]
        deg = [0] * n
        for i in range(n):
            row = problem[i]
            for j in range(i + 1, n):
                if row[j]:
                    edges.append((i, j))
                    adj[i].append(j)
                    adj[j].append(i)
                    deg[i] += 1
                    deg[j] += 1

        if not edges:
            return [1] * n

        # Bipartiteness check
        color_bip = [-1] * n
        is_bipartite = True
        for start in range(n):
            if color_bip[start] != -1:
                continue
            if not adj[start]:
                color_bip[start] = 0
                continue
            stack = [start]
            color_bip[start] = 0
            while stack:
                v = stack.pop()
                for u in adj[v]:
                    if color_bip[u] == -1:
                        color_bip[u] = 1 - color_bip[v]
                        stack.append(u)
                    elif color_bip[u] == color_bip[v]:
                        is_bipartite = False
                        break
                if not is_bipartite:
                    break
            if not is_bipartite:
                break

        if is_bipartite:
            return [c + 1 for c in color_bip]

        # DSatur for upper bound
        colors_greedy = self._dsatur(n, adj, deg)
        ub = max(colors_greedy)

        # Lower bound from clique
        adj_sets = [set(adj[i]) for i in range(n)]
        best_clique = self._greedy_clique(n, adj_sets, deg)
        lb = max(3, len(best_clique))

        if lb == ub:
            return colors_greedy

        # Try networkx clique for better bound
        G = nx.Graph()
        G.add_nodes_from(range(n))
        G.add_edges_from(edges)
        nx_clique = approx_clique.max_clique(G)
        if len(nx_clique) > lb:
            lb = len(nx_clique)
            best_clique = list(nx_clique)
            if lb == ub:
                return colors_greedy

        Q = sorted(best_clique)

        # Domination preprocessing
        active, dominator = self._domination_preprocess(n, adj_sets)
        V = sorted(active)
        V_set = active
        E_red = [(u, v) for u, v in edges if u in V_set and v in V_set]

        # Map clique to reduced graph
        Q_red = self._map_clique_to_reduced(Q, dominator, V_set, adj_sets)

        # Build reduced adjacency for potential exact solver
        adj_red = {v: [] for v in V}
        for u, v in E_red:
            adj_red[u].append(v)
            adj_red[v].append(u)

        # Try exact DSatur backtracking for small graphs
        nv = len(V)
        if nv <= 50:
            exact_result = self._dsatur_exact(V, adj_red, ub)
            if exact_result is not None:
                colors = [0] * n
                for v in range(n):
                    root = v
                    while dominator[root] != root:
                        root = dominator[root]
                    colors[v] = exact_result[root]
                used = sorted(set(colors))
                remap = {old: new for new, old in enumerate(used, start=1)}
                return [remap[c] for c in colors]

        # Decision version with SAT/CP-SAT for larger graphs
        for k in range(lb, ub):
            if HAS_PYSAT:
                result = self._try_color_sat(V, E_red, k, Q_red)
            else:
                result = self._try_color_cpsat(V, E_red, k, Q_red, colors_greedy)
            if result is not None:
                colors = [0] * n
                for v in range(n):
                    root = v
                    while dominator[root] != root:
                        root = dominator[root]
                    colors[v] = result[root]
                return colors

        return colors_greedy

    def _dsatur_exact(self, V, adj_red, ub):
        """Exact graph coloring using DSatur with backtracking."""
        n = len(V)
        deg_map = {v: len(adj_red[v]) for v in V}

        best_k = [ub]
        best_coloring = [None]
        color_assign = {}
        neighbor_colors = {v: set() for v in V}

        uncolored = set(V)
        saturation = {v: 0 for v in V}

        def pick_vertex():
            best = None
            best_sat = -1
            best_deg = -1
            for v in uncolored:
                s = saturation[v]
                d = deg_map[v]
                if s > best_sat or (s == best_sat and d > best_deg):
                    best = v
                    best_sat = s
                    best_deg = d
            return best

        def backtrack():
            if not uncolored:
                k = max(color_assign.values())
                if k < best_k[0]:
                    best_k[0] = k
                    best_coloring[0] = dict(color_assign)
                return

            v = pick_vertex()

            used = neighbor_colors[v]
            for c in range(1, best_k[0]):
                if c in used:
                    continue

                color_assign[v] = c
                uncolored.discard(v)

                affected = []
                for u in adj_red[v]:
                    if u in uncolored and c not in neighbor_colors[u]:
                        neighbor_colors[u].add(c)
                        saturation[u] += 1
                        affected.append(u)

                backtrack()

                # Undo
                del color_assign[v]
                uncolored.add(v)
                for u in affected:
                    neighbor_colors[u].discard(c)
                    saturation[u] -= 1

        backtrack()
        return best_coloring[0]

    def _map_clique_to_reduced(self, Q, dominator, V_set, adj_sets):
        Q_red = []
        seen = set()
        for v in Q:
            root = v
            while dominator[root] != root:
                root = dominator[root]
            if root in V_set and root not in seen:
                Q_red.append(root)
                seen.add(root)
        Q_red.sort()
        Q_verified = []
        for v in Q_red:
            if all(u in adj_sets[v] for u in Q_verified):
                Q_verified.append(v)
        return Q_verified

    def _greedy_clique(self, n, adj_sets, deg):
        best_clique = []
        sorted_by_deg = sorted(range(n), key=lambda x: -deg[x])
        for start_idx in range(min(20, n)):
            start = sorted_by_deg[start_idx]
            clique = [start]
            candidates = adj_sets[start].copy()
            while candidates:
                best_v = max(candidates, key=lambda v: len(adj_sets[v] & candidates))
                clique.append(best_v)
                candidates = candidates & adj_sets[best_v]
            if len(clique) > len(best_clique):
                best_clique = clique
        return best_clique

    def _dsatur(self, n, adj, degrees):
        colors = [0] * n
        colored = [False] * n
        saturation = [0] * n
        neighbor_colors = [set() for _ in range(n)]
        for _ in range(n):
            best = -1
            best_sat = -1
            best_deg = -1
            for v in range(n):
                if not colored[v]:
                    if (saturation[v] > best_sat or
                            (saturation[v] == best_sat and degrees[v] > best_deg)):
                        best = v
                        best_sat = saturation[v]
                        best_deg = degrees[v]
            v = best
            c = 1
            while c in neighbor_colors[v]:
                c += 1
            colors[v] = c
            colored[v] = True
            for u in adj[v]:
                if not colored[u] and c not in neighbor_colors[u]:
                    neighbor_colors[u].add(c)
                    saturation[u] += 1
        return colors

    def _domination_preprocess(self, n, adj_sets):
        active = set(range(n))
        dominator = list(range(n))
        changed = True
        while changed:
            changed = False
            nodes = sorted(active)
            to_remove = set()
            for i in range(len(nodes)):
                u = nodes[i]
                if u in to_remove:
                    continue
                adj_u = adj_sets[u] & active
                for j in range(i + 1, len(nodes)):
                    v = nodes[j]
                    if v in to_remove:
                        continue
                    adj_v = adj_sets[v] & active
                    if adj_u <= adj_v:
                        to_remove.add(u)
                        dominator[u] = v
                        changed = True
                        break
                    elif adj_v <= adj_u:
                        to_remove.add(v)
                        dominator[v] = u
                        changed = True
            active -= to_remove
        return active, dominator

    def _try_color_sat(self, V, E, k, Q):
        """Try to color with k colors using SAT solver."""
        v_idx_map = {v: i for i, v in enumerate(V)}

        def var(v_idx, c):
            return v_idx * k + c + 1

        clauses = []

        for v in V:
            vi = v_idx_map[v]
            clauses.append([var(vi, c) for c in range(k)])

        for v in V:
            vi = v_idx_map[v]
            for c1 in range(k):
                for c2 in range(c1 + 1, k):
                    clauses.append([-var(vi, c1), -var(vi, c2)])

        for u, v in E:
            ui = v_idx_map[u]
            vi = v_idx_map[v]
            for c in range(k):
                clauses.append([-var(ui, c), -var(vi, c)])

        for i, v in enumerate(Q):
            if i < k and v in v_idx_map:
                vi = v_idx_map[v]
                clauses.append([var(vi, i)])

        solver = Glucose4()
        for cl in clauses:
            solver.add_clause(cl)

        if solver.solve():
            model = solver.get_model()
            result = {}
            for v in V:
                vi = v_idx_map[v]
                for c in range(k):
                    if model[var(vi, c) - 1] > 0:
                        result[v] = c + 1
                        break
            solver.delete()
            return result

        solver.delete()
        return None

    def _try_color_cpsat(self, V, E, k, Q, colors_greedy):
        model = cp_model.CpModel()
        c = {}
        for v in V:
            c[v] = model.NewIntVar(0, k - 1, f'c_{v}')
        for u, v in E:
            model.Add(c[u] != c[v])
        clique_set = set()
        for i, v in enumerate(Q):
            if i < k and v in c:
                model.Add(c[v] == i)
                clique_set.add(v)
        for v in V:
            if v not in clique_set:
                hint_val = colors_greedy[v] - 1
                if hint_val < k:
                    model.AddHint(c[v], hint_val)
        solver = cp_model.CpSolver()
        solver.parameters.max_time_in_seconds = 60
        solver.parameters.num_workers = 1
        status = solver.Solve(model)
        if status in (cp_model.FEASIBLE, cp_model.OPTIMAL):
            result = {}
            for v in V:
                result[v] = solver.Value(c[v]) + 1
            return result
        return None