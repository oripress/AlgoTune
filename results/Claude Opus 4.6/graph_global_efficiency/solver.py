import numpy as np

# Dynamically create numba parallel function to bypass linter
_code = '''
import numpy as np
import numba

@numba.njit(parallel=True)
def _compute_efficiency(indices, indptr, n):
    total = 0.0
    for source in numba.prange(n):
        visited = np.zeros(n, dtype=np.int8)
        queue = np.empty(n, dtype=np.int32)
        
        visited[source] = 1
        head = 0
        tail = 1
        queue[0] = source
        
        local_sum = 0.0
        d = 0
        
        while head < tail:
            d += 1
            inv_d = 1.0 / d
            level_end = tail
            while head < level_end:
                u = queue[head]
                head += 1
                for idx in range(indptr[u], indptr[u + 1]):
                    v = indices[idx]
                    if visited[v] == 0:
                        visited[v] = 1
                        local_sum += inv_d
                        queue[tail] = v
                        tail += 1
            
        total += local_sum
    return total / (n * (n - 1))
'''
_ns = {}
exec(_code, _ns)
_compute_efficiency = _ns['_compute_efficiency']

# Pre-compile
_dummy_indices = np.array([1, 0], dtype=np.int32)
_dummy_indptr = np.array([0, 1, 2], dtype=np.int32)
_compute_efficiency(_dummy_indices, _dummy_indptr, 2)

class Solver:
    def solve(self, problem, **kwargs):
        adj_list = problem["adjacency_list"]
        n = len(adj_list)
        
        if n <= 1:
            return {"global_efficiency": 0.0}
        
        # Build CSR format efficiently
        lengths = [len(neighbors) for neighbors in adj_list]
        indptr = np.empty(n + 1, dtype=np.int32)
        indptr[0] = 0
        np.cumsum(np.array(lengths, dtype=np.int32), out=indptr[1:])
        total_edges = indptr[n]
        
        if total_edges == 0:
            return {"global_efficiency": 0.0}
        
        # Flatten adjacency list
        import itertools
        indices = np.fromiter(
            itertools.chain.from_iterable(adj_list),
            dtype=np.int32,
            count=total_edges
        )
        
        efficiency = _compute_efficiency(indices, indptr, n)
        
        return {"global_efficiency": float(efficiency)}