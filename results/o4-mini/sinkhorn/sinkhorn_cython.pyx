# cython: boundscheck=False, wraparound=False, cdivision=True
import numpy as np
cimport numpy as np

ctypedef np.float64_t DTYPE_t

def sinkhorn_cyth(np.ndarray[DTYPE_t, ndim=1] a not None,
                  np.ndarray[DTYPE_t, ndim=1] b not None,
                  np.ndarray[DTYPE_t, ndim=2] K not None,
                  int maxiter, double tol):
    cdef int n = a.shape[0]
    cdef int m = b.shape[0]
    cdef np.ndarray[DTYPE_t, ndim=1] u = np.empty(n, dtype=np.float64)
    cdef np.ndarray[DTYPE_t, ndim=1] v = np.empty(m, dtype=np.float64)
    cdef np.ndarray[DTYPE_t, ndim=1] u_prev = np.empty(n, dtype=np.float64)
    cdef np.ndarray[DTYPE_t, ndim=1] Kv = np.empty(n, dtype=np.float64)
    cdef np.ndarray[DTYPE_t, ndim=1] KTu = np.empty(m, dtype=np.float64)
    cdef int i, j, it
    cdef double s, d, maxd
    # initialize scaling vectors
    for i in range(n):
        u[i] = 1.0 / n
    for j in range(m):
        v[j] = 1.0 / m
    # sinkhorn iterations
    for it in range(maxiter):
        # copy u
        for i in range(n):
            u_prev[i] = u[i]
        # Kv = K @ v
        for i in range(n):
            s = 0.0
            for j in range(m):
                s += K[i, j] * v[j]
            Kv[i] = s
        # update u
        for i in range(n):
            u[i] = a[i] / Kv[i]
        # KTu = K.T @ u
        for j in range(m):
            s = 0.0
            for i in range(n):
                s += K[i, j] * u[i]
            KTu[j] = s
        # update v
        for j in range(m):
            v[j] = b[j] / KTu[j]
        # check convergence every 10 iterations
        if it % 10 == 0:
            maxd = 0.0
            for i in range(n):
                d = u[i] - u_prev[i]
                if d < 0.0:
                    d = -d
                if d > maxd:
                    maxd = d
            if maxd < tol:
                break
    # compute transport plan P
    cdef np.ndarray[DTYPE_t, ndim=2] P = np.empty((n, m), dtype=np.float64)
    for i in range(n):
        for j in range(m):
            P[i, j] = u[i] * K[i, j] * v[j]
    return P
# cython: boundscheck=False, wraparound=False, nonecheck=False, cdivision=True
import numpy as np
cimport numpy as cnp

def sinkhorn_cython(cnp.ndarray[cnp.double_t, ndim=1] a,
                    cnp.ndarray[cnp.double_t, ndim=1] b,
                    cnp.ndarray[cnp.double_t, ndim=2] K,
                    int maxiter,
                    double tol):
    cdef:
        int n = a.shape[0]
        int m = b.shape[0]
        int it, i, j
        cnp.ndarray[cnp.double_t, ndim=1] u = np.empty(n, dtype=np.float64)
        cnp.ndarray[cnp.double_t, ndim=1] v = np.empty(m, dtype=np.float64)
        cnp.ndarray[cnp.double_t, ndim=1] u_prev = np.empty(n, dtype=np.float64)
        cnp.ndarray[cnp.double_t, ndim=1] K_v = np.empty(n, dtype=np.float64)
        cnp.ndarray[cnp.double_t, ndim=1] KTu = np.empty(m, dtype=np.float64)
        cnp.ndarray[cnp.double_t, ndim=2] P = np.empty((n, m), dtype=np.float64)
        double acc, diff, maxdiff
    # initialize u, v
    for i in range(n):
        u[i] = 1.0 / n
    for j in range(m):
        v[j] = 1.0 / m
    for it in range(maxiter):
        # copy u to u_prev
        for i in range(n):
            u_prev[i] = u[i]
        # compute K_v = K dot v
        for i in range(n):
            acc = 0.0
            for j in range(m):
                acc += K[i, j] * v[j]
            K_v[i] = acc
        # update u
        for i in range(n):
            u[i] = a[i] / K_v[i]
        # compute KTu = K^T dot u
        for j in range(m):
            acc = 0.0
            for i in range(n):
                acc += K[i, j] * u[i]
            KTu[j] = acc
        # update v
        for j in range(m):
            v[j] = b[j] / KTu[j]
        # check convergence every 10 iterations
        if it % 10 == 0:
            maxdiff = 0.0
            for i in range(n):
                diff = u[i] - u_prev[i]
                if diff < 0:
                    diff = -diff
                if diff > maxdiff:
                    maxdiff = diff
            if maxdiff < tol:
                break
    # compute P = diag(u) K diag(v)
    for i in range(n):
        for j in range(m):
            P[i, j] = u[i] * K[i, j] * v[j]
    return P