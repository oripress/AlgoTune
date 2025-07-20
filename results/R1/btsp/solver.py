import numpy as np
import numba
from numba import njit

@njit(cache=True)
def optimized_dp(n, neighbors_arr, neighbor_counts, graph, candidate):
    # DP table: dp[mask][i] = parent node
    dp = -np.ones((1 << n, n), dtype=np.int32)
    # Start at node 0
    start_mask = 1 << 0
    dp[start_mask, 0] = 0
    full_mask = (1 << n) - 1
    
    # Precompute bit masks
    bit_masks = np.array([1 << i for i in range(n)], dtype=np.uint64)
    
    # Iterate over masks
    for mask in range(1, 1 << n):
        for i in range(n):
            if dp[mask, i] == -1:
                continue
                
            # If we've visited all nodes, check return to 0
            if mask == full_mask:
                if graph[i, 0] <= candidate:
                    # Reconstruct path
                    path = np.zeros(n + 1, dtype=np.int32)
                    path[0] = 0
                    path[n] = 0
                    curr = i
                    mask_state = mask
                    idx = n - 1
                    while curr != 0:
                        path[idx] = curr
                        idx -= 1
                        prev = dp[mask_state, curr]
                        mask_state = mask_state ^ bit_masks[curr]
                        curr = prev
                    return True, path
            else:
                # Process neighbors
                count = neighbor_counts[i]
                for idx in range(count):
                    j = neighbors_arr[i, idx]
                    bit = bit_masks[j]
                    if mask & bit:
                        continue
                    new_mask = mask | bit
                    if dp[new_mask, j] == -1:
                        dp[new_mask, j] = i
                        # Early termination if we reached full mask
                        if new_mask == full_mask and graph[j, 0] <= candidate:
                            # Reconstruct path
                            path = np.zeros(n + 1, dtype=np.int32)
                            path[0] = 0
                            path[n] = 0
                            curr = j
                            mask_state = new_mask
                            idx = n - 1
                            while curr != 0:
                                path[idx] = curr
                                idx -= 1
                                prev = dp[mask_state, curr]
                                mask_state = mask_state ^ bit_masks[curr]
                                curr = prev
                            return True, path
    
    # Check full_mask states
    for i in range(n):
        if dp[full_mask, i] != -1 and graph[i, 0] <= candidate:
            # Reconstruct path
            path = np.zeros(n + 1, dtype=np.int32)
            path[0] = 0
            path[n] = 0
            curr = i
            mask_state = full_mask
            idx = n - 1
            while curr != 0:
                path[idx] = curr
                idx -= 1
                prev = dp[mask_state, curr]
                mask_state = mask_state ^ bit_masks[curr]
                curr = prev
            return True, path
    
    return False, np.zeros(0, dtype=np.int32)

class Solver:
    def solve(self, problem, **kwargs):
        n = len(problem)
        if n == 0:
            return [0]
        if n == 1:
            return [0, 0]
        if n == 2:
            return [0, 1, 0]
        
        graph = np.array(problem, dtype=np.float64)
        edge_set = set()
        for i in range(n):
            for j in range(i+1, n):
                edge_set.add(problem[i][j])
        sorted_edges = sorted(edge_set)
        
        low, high = 0, len(sorted_edges) - 1
        best_cycle = None
        
        # Binary search for minimal bottleneck
        while low <= high:
            mid = (low + high) // 2
            candidate = sorted_edges[mid]
            exists, cycle = self.hamiltonian_cycle(graph, candidate, n)
            if exists:
                best_cycle = cycle
                high = mid - 1
            else:
                low = mid + 1
                
        if best_cycle is None:
            candidate = sorted_edges[-1]
            _, best_cycle = self.hamiltonian_cycle(graph, candidate, n)
            if best_cycle is None:
                return list(range(n)) + [0]
        
        return best_cycle

    def hamiltonian_cycle(self, graph, candidate, n):
        # Precompute neighbors as arrays
        neighbors_arr = -np.ones((n, n), dtype=np.int32)
        neighbor_counts = np.zeros(n, dtype=np.int32)
        
        for i in range(n):
            count = 0
            for j in range(n):
                if i != j and graph[i,j] <= candidate:
                    neighbors_arr[i, count] = j
                    count += 1
            neighbor_counts[i] = count
            if count < 2:
                return False, None
        
        exists, path = optimized_dp(n, neighbors_arr, neighbor_counts, graph, candidate)
        if exists:
            return True, path.tolist()
        return False, None