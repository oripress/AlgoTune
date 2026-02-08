import numpy as np
from scipy.optimize import linprog
from scipy.sparse import csr_matrix
from pysat.solvers import Minicard

class Solver:
    def solve(self, problem, **kwargs):
        n = len(problem)
        if n == 0:
            return []

        edges = []
        adj = [[] for _ in range(n)]
        for i in range(n):
            row = problem[i]
            for j in range(i + 1, n):
                if row[j]:
                    edges.append((i, j))
                    adj[i].append(j)
                    adj[j].append(i)

        if not edges:
            return []

        in_cover = []
        removed = [False] * n
        deg = [len(a) for a in adj]

        queue = [v for v in range(n) if deg[v] <= 1]
        while queue:
            v = queue.pop()
            if removed[v]:
                continue
            if deg[v] == 0:
                removed[v] = True
                continue
            if deg[v] == 1:
                u = None
                for w in adj[v]:
                    if not removed[w]:
                        u = w
                        break
                if u is None:
                    removed[v] = True
                    continue
                in_cover.append(u)
                removed[v] = True
                removed[u] = True
                for w in adj[u]:
                    if not removed[w]:
                        deg[w] -= 1
                        if deg[w] <= 1:
                            queue.append(w)
                for w in adj[v]:
                    if not removed[w]:
                        deg[w] -= 1
                        if deg[w] <= 1:
                            queue.append(w)

        remaining = [v for v in range(n) if not removed[v]]
        remaining_edges = [(i, j) for i, j in edges if not removed[i] and not removed[j]]

        if not remaining_edges:
            return sorted(in_cover)

        rn = len(remaining)
        idx_map = {v: k for k, v in enumerate(remaining)}
        mapped_edges = [(idx_map[i], idx_map[j]) for i, j in remaining_edges]

        # LP relaxation for Nemhauser-Trotter reduction
        me = len(mapped_edges)
        kernel = list(range(rn))
        kernel_edges = mapped_edges
        lp_lb = 0

        try:
            c = np.ones(rn)
            rows = np.empty(2 * me, dtype=np.int32)
            cols = np.empty(2 * me, dtype=np.int32)
            for k_idx, (i, j) in enumerate(mapped_edges):
                rows[2 * k_idx] = k_idx
                rows[2 * k_idx + 1] = k_idx
                cols[2 * k_idx] = i
                cols[2 * k_idx + 1] = j
            vals = np.full(2 * me, -1.0)
            A = csr_matrix((vals, (rows, cols)), shape=(me, rn))
            b = np.full(me, -1.0)
            bounds = [(0, 1)] * rn
            res = linprog(c, A_ub=A, b_ub=b, bounds=bounds, method='highs')

            if res.success:
                x = res.x
                lp_lb = int(np.ceil(res.fun - 1e-6))
                fixed_one = []
                undecided = []
                for i in range(rn):
                    if x[i] > 0.75:
                        fixed_one.append(i)
                    elif x[i] < 0.25:
                        pass
                    else:
                        undecided.append(i)

                for i in fixed_one:
                    in_cover.append(remaining[i])

                if not undecided:
                    return sorted(in_cover)

                und_set = set(undecided)
                k_edges_orig = [(i, j) for i, j in mapped_edges if i in und_set and j in und_set]
                if not k_edges_orig:
                    return sorted(in_cover)

                kidx = {v: k for k, v in enumerate(undecided)}
                kernel = undecided
                kernel_edges = [(kidx[i], kidx[j]) for i, j in k_edges_orig]
                lp_lb = max(0, lp_lb - len(fixed_one))
        except Exception:
            pass

        kn = len(kernel)
        ke = kernel_edges
        if not ke:
            return sorted(in_cover)

        matched = [False] * kn
        msz = 0
        for i, j in ke:
            if not matched[i] and not matched[j]:
                matched[i] = True
                matched[j] = True
                msz += 1
        lb = max(msz, lp_lb)

        lits = list(range(1, kn + 1))
        left = lb
        right = kn
        best = list(range(kn))

        with Minicard() as solver:
            for u, v in ke:
                solver.add_clause([u + 1, v + 1])
            solver.add_atmost(lits, left)
            if solver.solve():
                model = solver.get_model()
                best = [i for i in range(kn) if model[i] > 0]
                result = list(in_cover)
                for k_val in best:
                    result.append(remaining[kernel[k_val]])
                return sorted(result)

        while right > left + 1:
            mid = (right + left) // 2
            with Minicard() as solver:
                for u, v in ke:
                    solver.add_clause([u + 1, v + 1])
                solver.add_atmost(lits, mid)
                if solver.solve():
                    model = solver.get_model()
                    selected = [i for i in range(kn) if model[i] > 0]
                    best = selected
                    right = len(selected)
                else:
                    left = mid

        result = list(in_cover)
        for k_val in best:
            result.append(remaining[kernel[k_val]])
        return sorted(result)