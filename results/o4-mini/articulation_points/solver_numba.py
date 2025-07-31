import numpy as np
from numba import njit

@njit(cache=True)
def solve_nb(n, u_arr, v_arr):
    m = u_arr.shape[0]
    E = m * 2
    head = np.full(n, -1, np.int32)
    to = np.empty(E, np.int32)
    nxt = np.empty(E, np.int32)
    disc = np.full(n, -1, np.int32)
    low = np.empty(n, np.int32)
    parent = np.full(n, -1, np.int32)
    children = np.zeros(n, np.int32)
    ap = np.zeros(n, np.int32)
    idx = 0
    for i in range(m):
        a = u_arr[i]; b = v_arr[i]
        to[idx] = b; nxt[idx] = head[a]; head[a] = idx; idx += 1
        to[idx] = a; nxt[idx] = head[b]; head[b] = idx; idx += 1
    TIME = 0
    stack_u = np.empty(n + E, np.int32)
    stack_j = np.empty(n + E, np.int32)
    top = 0
    for start in range(n):
        if disc[start] != -1:
            continue
        disc[start] = TIME; low[start] = TIME; TIME += 1
        stack_u[top] = start; stack_j[top] = head[start]; top += 1
        while top > 0:
            top -= 1
            u = stack_u[top]; j = stack_j[top]
            if j != -1:
                stack_u[top] = u; stack_j[top] = nxt[j]; top += 1
                v = to[j]
                if disc[v] == -1:
                    parent[v] = u; children[u] += 1
                    disc[v] = TIME; low[v] = TIME; TIME += 1
                    stack_u[top] = v; stack_j[top] = head[v]; top += 1
                elif v != parent[u]:
                    if disc[v] < low[u]:
                        low[u] = disc[v]
            else:
                p = parent[u]
                if p != -1:
                    if low[u] < low[p]:
                        low[p] = low[u]
                    if parent[p] != -1 and low[u] >= disc[p]:
                        ap[p] = 1
                else:
                    if children[u] > 1:
                        ap[u] = 1
    # Return mask of articulation points (1 for AP, 0 otherwise)
    return ap

# Warm-up JIT compilation to pre-compile solve_nb
_dummy_u = np.empty(0, np.int32)
_dummy_v = np.empty(0, np.int32)
solve_nb(0, _dummy_u, _dummy_v)