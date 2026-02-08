import numpy as np
from numba import njit

@njit(cache=True)
def _solve_btsp(dist, n):
    if n <= 1:
        return np.array([0, 0], dtype=np.int32)
    if n == 2:
        return np.array([0, 1, 0], dtype=np.int32)
    
    full_mask = (1 << n) - 1
    size = 1 << n
    ne = n * (n - 1) // 2
    
    # Collect and sort edge weights
    weights = np.empty(ne, dtype=np.float64)
    k = 0
    for i in range(n):
        for j in range(i + 1, n):
            weights[k] = dist[i, j]
            k += 1
    weights.sort()
    
    # Deduplicate
    unique = np.empty(ne, dtype=np.float64)
    uc = 1
    unique[0] = weights[0]
    for i in range(1, ne):
        if weights[i] != weights[i - 1]:
            unique[uc] = weights[i]
            uc += 1
    
    # Compute MST bottleneck (lower bound)
    in_mst = np.zeros(n, dtype=np.bool_)
    in_mst[0] = True
    mst_key = np.full(n, 1e18)
    for j in range(1, n):
        mst_key[j] = dist[0, j]
    
    mst_bn = 0.0
    for _ in range(n - 1):
        u = -1
        mk = 1e18
        for j in range(n):
            if not in_mst[j] and mst_key[j] < mk:
                mk = mst_key[j]
                u = j
        in_mst[u] = True
        if mk > mst_bn:
            mst_bn = mk
        for j in range(n):
            if not in_mst[j] and dist[u, j] < mst_key[j]:
                mst_key[j] = dist[u, j]
    
    # Find starting lo from MST bottleneck
    lo = 0
    for i in range(uc):
        if unique[i] >= mst_bn - 1e-12:
            lo = i
            break
    
    adj = np.zeros(n, dtype=np.int64)
    dp = np.zeros(size, dtype=np.int64)
    
    hi = uc - 1
    
    while lo < hi:
        mid = (lo + hi) // 2
        T = unique[mid]
        
        # Build adjacency
        for i in range(n):
            adj[i] = 0
            for j in range(n):
                if i != j and dist[i, j] <= T:
                    adj[i] |= np.int64(1) << np.int64(j)
        
        # Degree check
        ok = True
        for i in range(n):
            t = adj[i]
            c = 0
            while t:
                c += 1
                t &= t - 1
            if c < 2:
                ok = False
                break
        
        if not ok:
            lo = mid + 1
            continue
        
        # DP check
        for i in range(size):
            dp[i] = 0
        dp[1] = np.int64(1)
        
        for mask in range(1, size):
            if not (mask & 1) or dp[mask] == 0:
                continue
            dpm = dp[mask]
            for u in range(1, n):
                bit_u = np.int64(1) << np.int64(u)
                if mask & bit_u:
                    continue
                if dpm & adj[u]:
                    dp[mask | bit_u] |= bit_u
        
        if dp[full_mask] & adj[0]:
            hi = mid
        else:
            lo = mid + 1
    
    # Build adjacency for optimal threshold
    T = unique[lo]
    for i in range(n):
        adj[i] = 0
        for j in range(n):
            if i != j and dist[i, j] <= T:
                adj[i] |= np.int64(1) << np.int64(j)
    
    # Final DP
    for i in range(size):
        dp[i] = 0
    dp[1] = np.int64(1)
    
    for mask in range(1, size):
        if not (mask & 1) or dp[mask] == 0:
            continue
        dpm = dp[mask]
        for u in range(1, n):
            bit_u = np.int64(1) << np.int64(u)
            if mask & bit_u:
                continue
            if dpm & adj[u]:
                dp[mask | bit_u] |= bit_u
    
    # Find endpoint
    endpoints = dp[full_mask] & adj[0]
    last = 0
    t = endpoints & (-endpoints)
    while t > 1:
        t >>= 1
        last += 1
    
    # Reconstruct
    path = np.zeros(n + 1, dtype=np.int32)
    path[0] = 0
    path[n] = 0
    mask = np.int64(full_mask)
    v = last
    for step in range(n - 1, 0, -1):
        path[step] = v
        prev_mask = mask ^ (np.int64(1) << np.int64(v))
        cands = dp[prev_mask] & adj[v]
        pred = 0
        t = cands & (-cands)
        while t > 1:
            t >>= 1
            pred += 1
        mask = prev_mask
        v = pred
    
    return path


class Solver:
    def __init__(self):
        # Warm up numba JIT
        d2 = np.array([[0.0, 1.0], [1.0, 0.0]])
        _solve_btsp(d2, 2)
        d4 = np.array([[0., 10., 20., 30.], [10., 0., 25., 35.],
                        [20., 25., 0., 15.], [30., 35., 15., 0.]])
        _solve_btsp(d4, 4)
    
    def solve(self, problem, **kwargs):
        n = len(problem)
        if n <= 1:
            return [0, 0]
        
        dist = np.array(problem, dtype=np.float64)
        
        if n <= 22:
            return _solve_btsp(dist, n).tolist()
        else:
            return self._solve_large(problem, n)
    
    def _solve_large(self, dist_list, n):
        from ortools.sat.python import cp_model
        
        weights = sorted(set(
            dist_list[i][j]
            for i in range(n)
            for j in range(i + 1, n)
        ))
        
        lo, hi = 0, len(weights) - 1
        best_tour = None
        
        while lo < hi:
            mid = (lo + hi) // 2
            T = weights[mid]
            tour = self._check_hc(dist_list, n, T)
            if tour is not None:
                best_tour = tour
                hi = mid
            else:
                lo = mid + 1
        
        if best_tour is None:
            best_tour = self._check_hc(dist_list, n, weights[lo])
        
        return best_tour
    
    def _check_hc(self, dist, n, threshold):
        from ortools.sat.python import cp_model
        
        model = cp_model.CpModel()
        arcs = []
        for i in range(n):
            for j in range(n):
                if i != j and dist[i][j] <= threshold + 1e-9:
                    lit = model.NewBoolVar(f'{i}_{j}')
                    arcs.append((i, j, lit))
        
        model.AddCircuit(arcs)
        solver = cp_model.CpSolver()
        solver.parameters.max_time_in_seconds = 30.0
        status = solver.Solve(model)
        
        if status in (cp_model.FEASIBLE, cp_model.OPTIMAL):
            nc = {}
            for i, j, lit in arcs:
                if solver.Value(lit):
                    nc[i] = j
            path = [0]
            c = 0
            for _ in range(n):
                c = nc[c]
                path.append(c)
            return path
        return None