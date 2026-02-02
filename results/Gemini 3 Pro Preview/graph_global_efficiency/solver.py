import numpy as np
from itertools import chain
import os
try:
    from fast_efficiency import calculate_efficiency_cython
except ImportError:
    # Fallback or compilation failed, though the harness should handle it.
    pass

class Solver:
    def solve(self, problem, **kwargs):
        adj_list = problem["adjacency_list"]
        n = len(adj_list)
        if n <= 1:
            return {"global_efficiency": 0.0}
        
        # Efficiently construct CSR matrix components
        lengths = np.array([len(x) for x in adj_list], dtype=np.int32)
        indptr = np.zeros(n + 1, dtype=np.int32)
        np.cumsum(lengths, out=indptr[1:])
        
        num_edges = indptr[-1]
        # Use fromiter to avoid creating a huge intermediate list
        indices = np.fromiter(chain.from_iterable(adj_list), dtype=np.int32, count=num_edges)
        
        # Determine number of threads
        # Use a reasonable default or check environment variables if needed
        # For this environment, we can try to use available CPUs
        try:
            num_threads = len(os.sched_getaffinity(0))
        except AttributeError:
            num_threads = os.cpu_count() or 4
            
        total_inv_dist = calculate_efficiency_cython(indptr, indices, n, num_threads)
            
        efficiency = total_inv_dist / (n * (n - 1))
        return {"global_efficiency": efficiency}