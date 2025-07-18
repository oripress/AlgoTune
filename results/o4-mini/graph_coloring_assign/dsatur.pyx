# cython: language_level=3, boundscheck=False, wraparound=False
from libc.stdlib cimport malloc, free
cdef extern from *:
    int __builtin_popcountll(unsigned long long x)
    int __builtin_ctzll(unsigned long long x)

cdef int MAXN = 128
cdef int N_GLOBAL

cdef int best_k
cdef unsigned char *best_sol
cdef unsigned char *colors
cdef unsigned long long *satur
cdef unsigned long long *neighbors_c
cdef int *degrees_c

cdef int changed_stack[MAXN][MAXN]

cdef void dfs(int depth, int colored_count, int current_max):
    cdef int i, u, sat, d, max_sat, max_deg, cc, idx, v, c, new_max
    cdef unsigned long long avail, mb, bit
    # Prune
    if current_max >= best_k:
        return
    # Complete
    if colored_count == N_GLOBAL:
        best_k = current_max
        for i in range(N_GLOBAL):
            best_sol[i] = colors[i]
        return
    # Select next vertex u
    max_sat = -1
    max_deg = -1
    u = -1
    for i in range(N_GLOBAL):
        if colors[i] == 0:
            sat = __builtin_popcountll(satur[i])
            d = degrees_c[i]
            if sat > max_sat or (sat == max_sat and d > max_deg):
                max_sat = sat
                max_deg = d
                u = i
    # Compute available colors bitmask
    if current_max + 1 < best_k:
        avail = (((<unsigned long long>1) << (current_max + 1)) - 1) & ~satur[u]
    else:
        avail = (((<unsigned long long>1) << best_k) - 1) & ~satur[u]
    cc = 0
    # Try each color
    while avail:
        bit = avail & -avail
        avail ^= bit
        c = __builtin_ctzll(bit) + 1
        colors[u] = <unsigned char> c
        new_max = c if c > current_max else current_max
        # Update satur and record changes
        mb = neighbors_c[u]
        while mb:
            v = __builtin_ctzll(mb)
            mb &= mb - 1
            if colors[v] == 0 and not (satur[v] & bit):
                satur[v] |= bit
                changed_stack[depth][cc] = v
                cc += 1
        dfs(depth+1, colored_count+1, new_max)
        # Backtrack
        for idx in range(cc-1, -1, -1):
            v = changed_stack[depth][idx]
            satur[v] ^= bit
        colors[u] = 0

def solve_dsatur(neighbors_py, degrees_py, greedy_py, clique_py, int UB):
    """
    neighbors_py: list of Python ints (bitmask)
    degrees_py: list of Python ints
    greedy_py: list of Python ints
    clique_py: list of Python ints
    UB: int
    """
    cdef int N, i, LB, colored
    global N_GLOBAL, best_k, best_sol, colors, satur, neighbors_c, degrees_c
    N = len(neighbors_py)
    if N > MAXN:
        raise ValueError("graph too large")
    N_GLOBAL = N
    # Allocate arrays
    colors = <unsigned char *> malloc(N * sizeof(unsigned char))
    satur = <unsigned long long *> malloc(N * sizeof(unsigned long long))
    neighbors_c = <unsigned long long *> malloc(N * sizeof(unsigned long long))
    degrees_c = <int *> malloc(N * sizeof(int))
    best_sol = <unsigned char *> malloc(N * sizeof(unsigned char))
    # Initialize and copy data
    for i in range(N):
        colors[i] = 0
        satur[i] = 0
        neighbors_c[i] = <unsigned long long> neighbors_py[i]
        degrees_c[i] = degrees_py[i]
        best_sol[i] = greedy_py[i]
    # Seed clique
    LB = len(clique_py)
    for i in range(LB):
        v = clique_py[i]
        c = i + 1
        colors[v] = <unsigned char> c
        bit = (<unsigned long long>1) << (c-1)
        mb = neighbors_c[v]
        while mb:
            j = __builtin_ctzll(mb)
            mb &= mb - 1
            if colors[j] == 0:
                satur[j] |= bit
    colored = LB
    best_k = UB
    dfs(LB, colored, LB)
    # Build Python list result
    res = [0] * N
    for i in range(N):
        res[i] = best_sol[i]
    # Free memory
    free(colors)
    free(satur)
    free(neighbors_c)
    free(degrees_c)
    free(best_sol)
    return res