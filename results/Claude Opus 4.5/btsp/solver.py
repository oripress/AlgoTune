import numpy as np
from numba import njit

@njit(cache=True)
def find_cycle_dp(adj, n):
    """DP-based Hamiltonian cycle finder for small n"""
    if n > 18:
        return np.array([-2], dtype=np.int64)  # Too large
    
    FULL = (1 << n) - 1
    
    # Check degrees first
    for i in range(n):
        deg = 0
        for j in range(n):
            if adj[i, j]:
                deg += 1
        if deg < 2:
            return np.array([-1], dtype=np.int64)
    
    # dp[mask] is bitmask of reachable last nodes with that visited set
    dp = np.zeros(1 << n, dtype=np.int64)
    parent = np.zeros((1 << n, n), dtype=np.int64)
    parent[:, :] = -1
    
    dp[1] = 1  # Start at node 0
    
    for mask in range(1, 1 << n):
        if dp[mask] == 0:
            continue
        for last in range(n):
            if not (dp[mask] & (1 << last)):
                continue
            for nxt in range(n):
                if mask & (1 << nxt):
                    continue
                if adj[last, nxt]:
                    new_mask = mask | (1 << nxt)
                    if not (dp[new_mask] & (1 << nxt)):
                        dp[new_mask] |= (1 << nxt)
                        parent[new_mask, nxt] = last

    # Find end node
    end_node = -1
    for last in range(1, n):
        if (dp[FULL] & (1 << last)) and adj[last, 0]:
            end_node = last
            break
    
    if end_node == -1:
        return np.array([-1], dtype=np.int64)
    
    # Reconstruct
    path = np.zeros(n + 1, dtype=np.int64)
    path[n] = 0
    mask = FULL
    current = end_node
    for i in range(n - 1, -1, -1):
        path[i] = current
        p = parent[mask, current]
        mask ^= (1 << current)
        current = p
    
    return path

@njit(cache=True)
def build_adj(problem, n, threshold):
    adj = np.zeros((n, n), dtype=np.bool_)
    for i in range(n):
        for j in range(n):
            if i != j and problem[i, j] <= threshold:
                adj[i, j] = True
    return adj

class Solver:
    def solve(self, problem, **kwargs):
        n = len(problem)
        if n <= 1:
            return [0, 0]
        if n == 2:
            return [0, 1, 0]
        
        problem_np = np.array(problem, dtype=np.float64)
        
        # Get sorted unique edge weights
        edge_weights = set()
        for i in range(n):
            for j in range(i+1, n):
                edge_weights.add(problem[i][j])
        
        sorted_weights = sorted(edge_weights)
        
        # Binary search for minimum bottleneck
        lo, hi = 0, len(sorted_weights) - 1
        best_tour = None
        
        while lo <= hi:
            mid = (lo + hi) // 2
            threshold = sorted_weights[mid]
            adj = build_adj(problem_np, n, threshold)
            result = find_cycle_dp(adj, n)
            
            if result[0] == -2:  # Too large for DP
                tour = self._find_tour_cpsat(problem, n, threshold)
                if tour:
                    best_tour = tour
                    hi = mid - 1
                else:
                    lo = mid + 1
            elif result[0] >= 0:
                best_tour = [int(x) for x in result]
                hi = mid - 1
            else:
                lo = mid + 1
        
        return best_tour if best_tour else list(range(n)) + [0]
    
    def _find_tour_cpsat(self, problem, n, threshold):
        from ortools.sat.python import cp_model
        
        # Check degree condition
        for i in range(n):
            deg = sum(1 for j in range(n) if i != j and problem[i][j] <= threshold)
            if deg < 2:
                return None
        
        model = cp_model.CpModel()
        arcs = []
        arc_vars = {}
        
        for i in range(n):
            for j in range(n):
                if i != j and problem[i][j] <= threshold:
                    var = model.NewBoolVar('')
                    arcs.append((i, j, var))
                    arc_vars[(i, j)] = var
        
        model.AddCircuit(arcs)
        
        solver = cp_model.CpSolver()
        solver.parameters.max_time_in_seconds = 2.0
        status = solver.Solve(model)
        
        if status in (cp_model.OPTIMAL, cp_model.FEASIBLE):
            next_city = {}
            for (i, j), var in arc_vars.items():
                if solver.Value(var):
                    next_city[i] = j
            
            tour = [0]
            current = 0
            for _ in range(n-1):
                current = next_city[current]
                tour.append(current)
            tour.append(0)
            return tour
        
        return None