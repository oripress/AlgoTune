import numpy as np
from numba import njit

@njit(nogil=True, cache=True)
def _global_eff(n, indptr, indices):
    total = 0.0
    queue = np.empty(n, np.int32)
    dist = np.empty(n, np.int32)
    for i in range(n):
        # reset distances
        for j in range(n):
            dist[j] = -1
        head = 0
        tail = 0
        dist[i] = 0
        queue[tail] = i
        tail += 1
        # BFS
        while head < tail:
            u = queue[head]
            head += 1
            du = dist[u]
            for k in range(indptr[u], indptr[u+1]):
                v = indices[k]
                if dist[v] < 0:
                    dist[v] = du + 1
                    queue[tail] = v
                    tail += 1
                    total += 1.0 / (du + 1)
    return total / (n * (n - 1))

# Pre-compile on a small dummy graph so first call to solve is fast
_dummy_indptr = np.zeros(3, np.int32)
_dummy_indices = np.empty(0, np.int32)
_global_eff(2, _dummy_indptr, _dummy_indices)

class Solver:
    def solve(self, problem, **kwargs):
        adj_list = problem.get("adjacency_list")
        if adj_list is None:
            return {"global_efficiency": 0.0}
        n = len(adj_list)
        if n <= 1:
            return {"global_efficiency": 0.0}

        # Flatten adjacency list to CSR format
        lengths = [len(neigh) for neigh in adj_list]
        nnz = sum(lengths)
        indptr = np.empty(n + 1, np.int32)
        indptr[0] = 0
        for i in range(n):
            indptr[i+1] = indptr[i] + lengths[i]

        indices = np.empty(nnz, np.int32)
        pos = 0
        for neigh in adj_list:
            for v in neigh:
                indices[pos] = v
                pos += 1

        eff = _global_eff(n, indptr, indices)
        return {"global_efficiency": float(eff)}