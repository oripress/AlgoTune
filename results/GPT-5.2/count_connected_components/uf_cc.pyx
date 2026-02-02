# cython: language_level=3
# cython: boundscheck=False
# cython: wraparound=False
# cython: cdivision=True
# cython: nonecheck=False

from cpython.mem cimport PyMem_Malloc, PyMem_Free
from cpython.sequence cimport PySequence_Fast
from cpython.object cimport PyObject

cdef extern from *:
    """
    #include <Python.h>

    static CYTHON_INLINE Py_ssize_t fast_seq_size(PyObject* seq) {
        return PySequence_Fast_GET_SIZE(seq);
    }
    static CYTHON_INLINE PyObject** fast_seq_items(PyObject* seq) {
        return PySequence_Fast_ITEMS(seq);
    }
    static CYTHON_INLINE PyObject* fast_tuple_get(PyObject* t, Py_ssize_t i) {
        return PyTuple_GET_ITEM(t, i);
    }
    static CYTHON_INLINE long fast_as_long(PyObject* o) {
        return PyLong_AS_LONG(o);
    }
    """
    Py_ssize_t fast_seq_size(PyObject* seq)
    PyObject** fast_seq_items(PyObject* seq)
    PyObject* fast_tuple_get(PyObject* t, Py_ssize_t i)
    long fast_as_long(PyObject* o)

cdef inline int _find(int* parent, int x):
    cdef int px
    cdef int ppx
    while True:
        px = parent[x]
        if px == x:
            return x
        ppx = parent[px]
        parent[x] = ppx
        x = px

def count_cc(int n, object edges):
    cdef int cc = n
    if n <= 1:
        return n
    if not edges:
        return n

    cdef int* parent = <int*>PyMem_Malloc(n * sizeof(int))
    cdef unsigned char* rank = <unsigned char*>PyMem_Malloc(n * sizeof(unsigned char))
    if parent == NULL or rank == NULL:
        if parent != NULL:
            PyMem_Free(parent)
        if rank != NULL:
            PyMem_Free(rank)
        raise MemoryError()

    cdef int i
    for i in range(n):
        parent[i] = i
        rank[i] = 0

    # Normalize edges to a "fast sequence" once; then iterate via raw pointer array.
    cdef object seq_obj = PySequence_Fast(edges, b"edges must be a sequence")
    cdef PyObject* seq = <PyObject*>seq_obj
    cdef Py_ssize_t m = fast_seq_size(seq)
    cdef PyObject** items = fast_seq_items(seq)

    cdef PyObject* e
    cdef int u, v, ru, rv, pu, pv
    cdef unsigned char rru, rrv

    for i in range(m):
        e = items[i]
        u = <int>fast_as_long(fast_tuple_get(e, 0))
        v = <int>fast_as_long(fast_tuple_get(e, 1))

        pu = parent[u]
        pv = parent[v]
        if pu == pv:
            continue

        # If node is already a root, avoid calling _find.
        ru = u if pu == u else _find(parent, u)
        rv = v if pv == v else _find(parent, v)
        if ru == rv:
            continue

        rru = rank[ru]
        rrv = rank[rv]
        if rru < rrv:
            parent[ru] = rv
        elif rru > rrv:
            parent[rv] = ru
        else:
            parent[rv] = ru
            rank[ru] = rru + 1

        cc -= 1
        if cc == 1:
            break

    PyMem_Free(parent)
    PyMem_Free(rank)
    return cc