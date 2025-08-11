import numpy as np
import numba as nb

@nb.njit
def _compute_all(data, indices, indptr, sources):
    n = indptr.shape[0] - 1
    k = sources.shape[0]
    res = np.empty((k, n), dtype=np.float64)
    for si in range(k):
        src = sources[si]
        dist = res[si]
        # initialize distances
        for i in range(n):
            dist[i] = np.inf
        visited = np.zeros(n, dtype=np.uint8)
        heap = np.empty(n, dtype=np.int64)
        pos = np.full(n, -1, dtype=np.int64)
        dist[src] = 0.0
        heap[0] = src
        pos[src] = 0
        size = 1
        while size > 0:
            u = heap[0]
            size -= 1
            if size > 0:
                last = heap[size]
                heap[0] = last
                pos[last] = 0
                i0 = 0
                while True:
                    left = 2 * i0 + 1
                    right = left + 1
                    smallest = i0
                    if left < size and dist[heap[left]] < dist[heap[smallest]]:
                        smallest = left
                    if right < size and dist[heap[right]] < dist[heap[smallest]]:
                        smallest = right
                    if smallest != i0:
                        tmp = heap[i0]
                        heap[i0] = heap[smallest]
                        heap[smallest] = tmp
                        pos[heap[i0]] = i0
                        pos[heap[smallest]] = smallest
                        i0 = smallest
                    else:
                        break
            if visited[u] == 1:
                continue
            visited[u] = 1
            du = dist[u]
            start = indptr[u]
            end = indptr[u+1]
            for j in range(start, end):
                v = indices[j]
                nd = du + data[j]
                if nd < dist[v]:
                    dist[v] = nd
                    p = pos[v]
                    if p != -1 and p < size:
                        ci = p
                        while ci > 0:
                            pa = (ci - 1) // 2
                            if dist[heap[ci]] < dist[heap[pa]]:
                                tmp2 = heap[pa]
                                heap[pa] = heap[ci]
                                heap[ci] = tmp2
                                pos[heap[pa]] = pa
                                pos[heap[ci]] = ci
                                ci = pa
                            else:
                                break
                    else:
                        ci = size
                        heap[ci] = v
                        pos[v] = ci
                        size += 1
                        while ci > 0:
                            pa = (ci - 1) // 2
                            if dist[heap[ci]] < dist[heap[pa]]:
                                tmp2 = heap[pa]
                                heap[pa] = heap[ci]
                                heap[ci] = tmp2
                                pos[heap[pa]] = pa
                                pos[heap[ci]] = ci
                                ci = pa
                            else:
                                break
        # next source
    return res

class Solver:
    def __init__(self):
        # Warm-up compilation
        _compute_all(np.array([0.0], dtype=np.float64),
                     np.array([0], dtype=np.int64),
                     np.array([0,0], dtype=np.int64),
                     np.array([0], dtype=np.int64))

    def solve(self, problem, **kwargs):
        try:
            data = np.array(problem["data"], dtype=np.float64)
            indices = np.array(problem["indices"], dtype=np.int64)
            indptr = np.array(problem["indptr"], dtype=np.int64)
            sources = np.array(problem["source_indices"], dtype=np.int64)
            if sources.size == 0:
                return {"distances": []}
            res = _compute_all(data, indices, indptr, sources)
            # Convert to Python lists (with inf for unreachable)
            return {"distances": res.tolist()}
        except Exception:
            return {"distances": []}