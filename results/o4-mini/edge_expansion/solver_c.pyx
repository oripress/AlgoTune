# cython: boundscheck=False, wraparound=False, cdivision=True
def compute_expansion(al, S):
    """
    Cython optimized edge expansion: |E(S, V-S)| / min(|S|, |V-S|)
    """
    cdef Py_ssize_t n = len(al)
    cdef Py_ssize_t s = len(S)
    # Edge cases
    if s == 0 or s == n:
        return 0.0
    # Create membership mask list
    cdef list m = [0] * n
    cdef Py_ssize_t i, j, boundary = 0
    cdef int u, v
    cdef list row
    # Mark S in mask
    for i in range(s):
        u = S[i]
        m[u] = 1
    # Count edges from S to V-S
    for i in range(s):
        u = S[i]
        row = al[u]
        for j in range(len(row)):
            v = row[j]
            if m[v] == 0:
                boundary += 1
    # Denominator
    cdef Py_ssize_t n_minus_s = n - s
    cdef Py_ssize_t denom = s if s <= n_minus_s else n_minus_s
    # Compute float result
    cdef double result = (<double>boundary) / denom
    return result