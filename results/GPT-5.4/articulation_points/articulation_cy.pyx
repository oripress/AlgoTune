# cython: language_level=3, boundscheck=False, wraparound=False, initializedcheck=False, cdivision=True

from libc.stdlib cimport malloc, free, realloc
from libc.string cimport memset

cdef class APSolver:
    cdef Py_ssize_t cap_n
    cdef Py_ssize_t cap_m2

    cdef Py_ssize_t *deg
    cdef Py_ssize_t *offsets
    cdef Py_ssize_t *fill
    cdef Py_ssize_t *neighbors
    cdef Py_ssize_t *disc
    cdef Py_ssize_t *low
    cdef Py_ssize_t *parent
    cdef unsigned char *is_ap
    cdef Py_ssize_t *stack_nodes
    cdef Py_ssize_t *stack_pos

    def __cinit__(self):
        self.cap_n = 0
        self.cap_m2 = 0

        self.deg = NULL
        self.offsets = NULL
        self.fill = NULL
        self.neighbors = NULL
        self.disc = NULL
        self.low = NULL
        self.parent = NULL
        self.is_ap = NULL
        self.stack_nodes = NULL
        self.stack_pos = NULL

    def __dealloc__(self):
        if self.deg != NULL:
            free(self.deg)
        if self.offsets != NULL:
            free(self.offsets)
        if self.fill != NULL:
            free(self.fill)
        if self.neighbors != NULL:
            free(self.neighbors)
        if self.disc != NULL:
            free(self.disc)
        if self.low != NULL:
            free(self.low)
        if self.parent != NULL:
            free(self.parent)
        if self.is_ap != NULL:
            free(self.is_ap)
        if self.stack_nodes != NULL:
            free(self.stack_nodes)
        if self.stack_pos != NULL:
            free(self.stack_pos)

    cdef void _ensure_n(self, Py_ssize_t n) except *:
        cdef Py_ssize_t new_cap
        cdef Py_ssize_t *tmp_i
        cdef unsigned char *tmp_b

        if n <= self.cap_n:
            return

        new_cap = self.cap_n if self.cap_n > 0 else 8
        while new_cap < n:
            new_cap <<= 1

        tmp_i = <Py_ssize_t*> realloc(self.deg, new_cap * sizeof(Py_ssize_t))
        if tmp_i == NULL:
            raise MemoryError()
        self.deg = tmp_i

        tmp_i = <Py_ssize_t*> realloc(self.offsets, (new_cap + 1) * sizeof(Py_ssize_t))
        if tmp_i == NULL:
            raise MemoryError()
        self.offsets = tmp_i

        tmp_i = <Py_ssize_t*> realloc(self.fill, new_cap * sizeof(Py_ssize_t))
        if tmp_i == NULL:
            raise MemoryError()
        self.fill = tmp_i

        tmp_i = <Py_ssize_t*> realloc(self.disc, new_cap * sizeof(Py_ssize_t))
        if tmp_i == NULL:
            raise MemoryError()
        self.disc = tmp_i

        tmp_i = <Py_ssize_t*> realloc(self.low, new_cap * sizeof(Py_ssize_t))
        if tmp_i == NULL:
            raise MemoryError()
        self.low = tmp_i

        tmp_i = <Py_ssize_t*> realloc(self.parent, new_cap * sizeof(Py_ssize_t))
        if tmp_i == NULL:
            raise MemoryError()
        self.parent = tmp_i

        tmp_b = <unsigned char*> realloc(self.is_ap, new_cap * sizeof(unsigned char))
        if tmp_b == NULL:
            raise MemoryError()
        self.is_ap = tmp_b

        tmp_i = <Py_ssize_t*> realloc(self.stack_nodes, new_cap * sizeof(Py_ssize_t))
        if tmp_i == NULL:
            raise MemoryError()
        self.stack_nodes = tmp_i

        tmp_i = <Py_ssize_t*> realloc(self.stack_pos, new_cap * sizeof(Py_ssize_t))
        if tmp_i == NULL:
            raise MemoryError()
        self.stack_pos = tmp_i

        self.cap_n = new_cap

    cdef void _ensure_m2(self, Py_ssize_t m2) except *:
        cdef Py_ssize_t new_cap
        cdef Py_ssize_t *tmp_i

        if m2 <= self.cap_m2:
            return

        new_cap = self.cap_m2 if self.cap_m2 > 0 else 16
        while new_cap < m2:
            new_cap <<= 1

        tmp_i = <Py_ssize_t*> realloc(self.neighbors, new_cap * sizeof(Py_ssize_t))
        if tmp_i == NULL:
            raise MemoryError()
        self.neighbors = tmp_i

        self.cap_m2 = new_cap

    cpdef list articulation_points(self, Py_ssize_t n, object edges):
        cdef Py_ssize_t m = len(edges)
        cdef Py_ssize_t i, u, v, j, t, sp, idx, pos, end, root, root_children, p, lu, dv
        cdef object e
        cdef list res

        if n <= 2 or m == 0:
            return []

        self._ensure_n(n)
        self._ensure_m2(2 * m)

        memset(self.deg, 0, n * sizeof(Py_ssize_t))
        memset(self.is_ap, 0, n * sizeof(unsigned char))

        for i in range(m):
            e = edges[i]
            u = e[0]
            v = e[1]
            self.deg[u] += 1
            self.deg[v] += 1

        self.offsets[0] = 0
        for i in range(n):
            self.offsets[i + 1] = self.offsets[i] + self.deg[i]
            self.fill[i] = self.offsets[i]
            self.disc[i] = -1
            self.parent[i] = -1

        for i in range(m):
            e = edges[i]
            u = e[0]
            v = e[1]

            j = self.fill[u]
            self.neighbors[j] = v
            self.fill[u] = j + 1

            j = self.fill[v]
            self.neighbors[j] = u
            self.fill[v] = j + 1

        t = 0
        for root in range(n):
            if self.disc[root] != -1:
                continue

            self.disc[root] = t
            self.low[root] = t
            t += 1
            root_children = 0

            sp = 1
            self.stack_nodes[0] = root
            self.stack_pos[0] = self.offsets[root]

            while sp > 0:
                idx = sp - 1
                u = self.stack_nodes[idx]
                pos = self.stack_pos[idx]
                end = self.offsets[u + 1]

                if pos < end:
                    v = self.neighbors[pos]
                    self.stack_pos[idx] = pos + 1

                    if self.disc[v] == -1:
                        self.parent[v] = u
                        if u == root:
                            root_children += 1
                        self.disc[v] = t
                        self.low[v] = t
                        t += 1
                        self.stack_nodes[sp] = v
                        self.stack_pos[sp] = self.offsets[v]
                        sp += 1
                    elif v != self.parent[u]:
                        dv = self.disc[v]
                        if dv < self.low[u]:
                            self.low[u] = dv
                else:
                    sp -= 1
                    p = self.parent[u]
                    if p != -1:
                        lu = self.low[u]
                        if lu < self.low[p]:
                            self.low[p] = lu
                        if lu >= self.disc[p]:
                            self.is_ap[p] = 1

            self.is_ap[root] = 1 if root_children > 1 else 0

        res = []
        for i in range(n):
            if self.is_ap[i]:
                res.append(i)
        return res